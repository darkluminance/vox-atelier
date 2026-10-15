from __future__ import annotations
import sys, json, io, wave, uuid, re, traceback, copy, os
from pathlib import Path

PROTOCOL = sys.stdout
sys.stdout = sys.stderr

def send(**message):
    PROTOCOL.write(json.dumps(message,ensure_ascii=True)+'\n'); PROTOCOL.flush()

def wav_bytes(audio, sample_rate):
    import numpy as np
    if hasattr(audio,'detach'): audio=audio.detach().float().cpu().numpy()
    data=np.asarray(audio,dtype=np.float32).squeeze()
    if data.ndim>1: data=data[0]
    data=np.clip(data,-1,1)
    pcm=(data*32767).astype('<i2').tobytes()
    out=io.BytesIO()
    with wave.open(out,'wb') as file:
        file.setnchannels(1); file.setsampwidth(2); file.setframerate(int(sample_rate)); file.writeframes(pcm)
    return out.getvalue(),pcm

def segments(text, mode):
    text=text.strip()
    if mode=='full' and len(text)<550: return [text]
    pieces=re.split(r'(?<=[.!?])\s+|\n{2,}',text)
    output=[]
    for piece in pieces:
        if not piece.strip(): continue
        if len(piece)<550: output.append(piece.strip())
        else:
            words=piece.split(); current=''
            for word in words:
                if len(current)+len(word)>500 and current: output.append(current); current=''
                current=(current+' '+word).strip()
            if current: output.append(current)
    return output or [text]

class Runtime:
    def __init__(self,item):
        self.item=item; self.family=item['family']; self.control=item['control']; self.path=item['path']; self.profile_key=None; self.prompt=None
        import torch
        self.device=os.getenv('TTS_DEVICE_'+self.family.upper()) or ('cuda' if torch.cuda.is_available() else 'cpu')
        if self.family=='qwen':
            from qwen_tts import Qwen3TTSModel
            dtype=torch.bfloat16 if self.device=='cuda' else torch.float32
            self.model=Qwen3TTSModel.from_pretrained(self.path,device_map=self.device,torch_dtype=dtype)
        elif self.family=='chatterbox':
            if self.control=='exaggeration':
                from chatterbox.tts import ChatterboxTTS
                self.model=ChatterboxTTS.from_local(self.path,device=self.device)
            else:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model=ChatterboxTurboTTS.from_local(self.path,device=self.device,nano=self.item['id'].endswith('nano'))
            self.default_conds=copy.deepcopy(self.model.conds)
        elif self.family=='csm':
            from transformers import AutoProcessor, CsmForConditionalGeneration
            self.processor=AutoProcessor.from_pretrained(self.item['processor_path'],local_files_only=True)
            self.model=CsmForConditionalGeneration.from_pretrained(self.path,torch_dtype=torch.bfloat16 if self.device=='cuda' else torch.float32,local_files_only=True).to(self.device)
        else: raise ValueError('Unsupported model family')

    def prepare(self,request):
        profile=request.get('profile')
        key=profile['id'] if profile else ''
        if self.family=='qwen' and self.control=='reference':
            if not profile: raise ValueError('Select a saved reference voice for Qwen Base')
            if not profile.get('transcript'): raise ValueError('Qwen Base requires the reference audio transcript')
            if key!=self.profile_key:
                self.prompt=self.model.create_voice_clone_prompt(ref_audio=profile['path'],ref_text=profile['transcript'])
                self.profile_key=key
        elif self.family=='chatterbox' and key!=self.profile_key:
            if profile:
                if self.control=='tags':
                    import soundfile as sf
                    if sf.info(profile['path']).duration<=5:
                        raise ValueError('Chatterbox Turbo and Nano need a reference WAV longer than 5 seconds')
                kwargs={'exaggeration':request.get('exaggeration',.5)} if self.control=='exaggeration' else {}
                self.model.prepare_conditionals(profile['path'],**kwargs)
            else: self.model.conds=copy.deepcopy(self.default_conds)
            self.profile_key=key

    def synthesize(self,text,request):
        import torch
        with torch.inference_mode():
            if self.family=='qwen':
                opts={'text':text,'language':'English'}
                if self.control=='named':
                    opts['speaker']=request.get('voice') or 'Ryan'
                    if self.item['id']=='qwen-17-custom' and request.get('style','').strip(): opts['instruct']=request['style'].strip()
                    result,sr=self.model.generate_custom_voice(**opts)
                elif self.control=='design':
                    opts['instruct']=request.get('style','').strip() or 'A clear, natural English speaking voice.'
                    result,sr=self.model.generate_voice_design(**opts)
                else:
                    opts['voice_clone_prompt']=self.prompt
                    result,sr=self.model.generate_voice_clone(**opts)
                return result[0],sr
            if self.family=='chatterbox':
                if self.control=='exaggeration':
                    audio=self.model.generate(text,exaggeration=float(request.get('exaggeration',.5)),cfg_weight=float(request.get('cfg_weight',.5)),temperature=float(request.get('temperature',.8)),top_p=float(request.get('top_p',.9)),repetition_penalty=float(request.get('repetition_penalty',1.2)))
                else:
                    audio=self.model.generate(text,temperature=float(request.get('temperature',.8)),top_k=int(request.get('top_k',100)),top_p=float(request.get('top_p',.9)),repetition_penalty=float(request.get('repetition_penalty',1.2)))
                return audio,self.model.sr
            if self.family=='csm':
                speaker=request.get('voice','0')
                profile=request.get('profile')
                if profile:
                    if not profile.get('transcript'): raise ValueError('CSM reference voice requires its spoken transcript')
                    import soundfile as sf
                    import librosa
                    reference,ref_rate=sf.read(profile['path'],dtype='float32')
                    if reference.ndim>1: reference=reference.mean(axis=1)
                    if ref_rate!=24000: reference=librosa.resample(reference,orig_sr=ref_rate,target_sr=24000)
                    conversation=[
                        {'role':speaker,'content':[{'type':'text','text':profile['transcript']},{'type':'audio','path':reference}]},
                        {'role':speaker,'content':[{'type':'text','text':text}]},
                    ]
                    inputs=self.processor.apply_chat_template(conversation,tokenize=True,return_dict=True).to(self.device)
                    inputs['input_values']=inputs['input_values'].to(dtype=next(self.model.parameters()).dtype)
                else:
                    inputs=self.processor(f'[{speaker}]{text}',add_special_tokens=True,return_tensors='pt').to(self.device)
                result=self.model.generate(**inputs,output_audio=True)
                audio=result.audio_values if hasattr(result,'audio_values') else result
                if isinstance(audio,list): audio=audio[0]
                return audio,self.model.config.codec_config.sampling_rate
        raise RuntimeError('Unknown model')

runtime=None
for line in sys.stdin:
    try:
        request=json.loads(line); kind=request.get('type')
        if kind=='quit': break
        if kind=='load':
            runtime=Runtime(request['item']); send(type='loaded',device=runtime.device); continue
        if kind=='generate':
            if not runtime: raise RuntimeError('Load a model first')
            runtime.prepare(request)
            parts=segments(request['text'],request['mode'])
            pcm_parts=[]; rate=None
            for index,part in enumerate(parts):
                audio,sample_rate=runtime.synthesize(part,request)
                chunk,pcm=wav_bytes(audio,sample_rate)
                if rate is not None and rate!=sample_rate: raise RuntimeError('Sample rate changed between chunks')
                rate=sample_rate; pcm_parts.append(pcm)
                if request['mode']=='stream':
                    chunk_path=Path(request['output_dir'])/(uuid.uuid4().hex+'.wav')
                    chunk_path.write_bytes(chunk)
                    send(type='chunk',index=index+1,total=len(parts),text=part,file=str(chunk_path))
                else: send(type='progress',index=index+1,total=len(parts))
            filename=uuid.uuid4().hex+'.wav'; path=Path(request['output_dir'])/filename
            with wave.open(str(path),'wb') as file:
                file.setnchannels(1); file.setsampwidth(2); file.setframerate(int(rate)); file.writeframes(b''.join(pcm_parts))
            send(type='done',file=str(path),segments=len(parts),sample_rate=rate)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        send(type='error',message=f'{type(error).__name__}: {error}')
