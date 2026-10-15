from __future__ import annotations
import asyncio, json, os, sys, uuid, base64, re
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
GENERATED = DATA / 'generated'
PROFILES = DATA / 'profiles'
for folder in (GENERATED, PROFILES): folder.mkdir(parents=True, exist_ok=True)
MODEL_ROOT = Path(os.getenv('TTS_MODEL_ROOT', 'D:/models'))
HF = MODEL_ROOT / 'huggingface/hub'
QWEN = MODEL_ROOT / 'qwen3-tts/checkpoints'
CSM_PROCESSOR = MODEL_ROOT / 'csm-1b/processor'
CSM_WEIGHTS = MODEL_ROOT / 'csm-1b/transformers'

def snapshot(repo: str) -> Path | None:
    root = HF / repo / 'snapshots'
    if not root.exists(): return None
    paths = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    return paths[0] if paths else None

def catalog():
    definitions = [
        ('qwen-17-custom','Qwen3 1.7B CustomVoice','qwen','Qwen3-TTS-12Hz-1.7B-CustomVoice','named','Style instruction'),
        ('qwen-17-design','Qwen3 1.7B VoiceDesign','qwen','Qwen3-TTS-12Hz-1.7B-VoiceDesign','design','Voice design prompt'),
        ('qwen-17-base','Qwen3 1.7B Base','qwen','Qwen3-TTS-12Hz-1.7B-Base','reference','Saved reference voice'),
        ('qwen-06-custom','Qwen3 0.6B CustomVoice','qwen','Qwen3-TTS-12Hz-0.6B-CustomVoice','named','Named speaker'),
        ('qwen-06-base','Qwen3 0.6B Base','qwen','Qwen3-TTS-12Hz-0.6B-Base','reference','Saved reference voice'),
        ('chatterbox','Chatterbox Original','chatterbox','models--ResembleAI--chatterbox','exaggeration','Expressiveness slider'),
        ('chatterbox-turbo','Chatterbox Turbo','chatterbox','models--ResembleAI--chatterbox-turbo','tags','Speech tags'),
        ('chatterbox-nano','Chatterbox Nano','chatterbox','models--ResembleAI--chatterbox-nano','tags','Speech tags'),
        ('csm-1b','Sesame CSM 1B','csm','models--sesame--csm-1b','context','Conversation speaker'),
    ]
    items=[]
    for id,name,family,folder,control,description in definitions:
        path=(QWEN / folder) if family=='qwen' else snapshot(folder)
        found=bool(path and path.exists())
        processor_path = path if family!='csm' else (path if path and (path/'preprocessor_config.json').exists() and (path/'tokenizer.json').exists() else CSM_PROCESSOR)
        csm_ready = (processor_path/'preprocessor_config.json').exists() and (processor_path/'tokenizer.json').exists() and (CSM_WEIGHTS/'transformers.safetensors.index.json').exists() and (CSM_WEIGHTS/'transformers-00001-of-00002.safetensors').exists() and (CSM_WEIGHTS/'transformers-00002-of-00002.safetensors').exists()
        ready=found and (family!='csm' or csm_ready)
        if family=='csm' and csm_ready: path=CSM_WEIGHTS
        items.append(dict(id=id,name=name,family=family,path=str(path) if path else '',processor_path=str(processor_path),found=found,assets_ready=bool(ready),control=control,description=description))
    return items

app=FastAPI(title='Vox Atelier')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['*'],allow_headers=['*'])
state={'proc':None,'model':None,'error':None,'log':None}
lock=asyncio.Lock()

class LoadBody(BaseModel): model_id:str
class GenerateBody(BaseModel): model_id:str; text:str; mode:str='full'; voice:str='Ryan'; style:str=''; profile_id:str=''; exaggeration:float=.5; cfg_weight:float=.5; temperature:float=.8; top_p:float=.9; top_k:int=100; repetition_penalty:float=1.2
class ProfileBody(BaseModel): name:str; transcript:str=''; audio_base64:str

async def stop_worker():
    proc=state['proc']
    if proc:
        try:
            proc.stdin.write(b'{"type":"quit"}\n'); await proc.stdin.drain()
            await asyncio.wait_for(proc.wait(),timeout=2)
        except Exception:
            proc.kill(); await proc.wait()
    if state['log']:
        state['log'].close()
    state.update(proc=None,model=None,log=None)

@app.on_event('shutdown')
async def shutdown_worker():
    await stop_worker()

async def start_worker(model_id:str):
    item=next((x for x in catalog() if x['id']==model_id),None)
    if not item: raise HTTPException(404,'Unknown model')
    if not item['found']: raise HTTPException(400,'Checkpoint not found on disk')
    if not item['assets_ready']: raise HTTPException(400,'Required local model assets are missing or incomplete')
    await stop_worker()
    env=os.environ.copy(); env['HF_HOME']=str(MODEL_ROOT/'huggingface'); env['HF_HUB_OFFLINE']='1'; env['TRANSFORMERS_OFFLINE']='1'
    python=env.get('TTS_PYTHON_'+item['family'].upper(),sys.executable)
    log=open(DATA/'worker.log','ab',buffering=0)
    try:
        proc=await asyncio.create_subprocess_exec(python,str(ROOT/'backend/worker.py'),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=log,env=env)
    except Exception as e:
        log.close()
        raise HTTPException(500,f'Could not start {python}: {e}')
    state.update(proc=proc,model=model_id,error=None,log=log)
    proc.stdin.write((json.dumps({'type':'load','item':item})+'\n').encode()); await proc.stdin.drain()
    try:
        line=await asyncio.wait_for(proc.stdout.readline(),timeout=300)
        message=json.loads(line) if line else {'type':'error','message':'Worker exited during load'}
    except Exception as e:
        await stop_worker(); raise HTTPException(500,f'Model load timed out or failed: {e}')
    if message.get('type')!='loaded':
        err=message.get('message','Model load failed')
        await stop_worker(); state['error']=err
        raise HTTPException(500,err)
    return {'model_id':model_id,'status':'loaded','device':message.get('device')}

@app.get('/api/models')
def models():
    proc=state['proc']
    loaded=state['model'] if proc and proc.returncode is None else None
    return {'models':catalog(),'loaded':loaded,'error':state['error']}

@app.post('/api/load')
async def load(body:LoadBody):
    async with lock:
        if state['model']==body.model_id and state['proc'] and state['proc'].returncode is None: return {'model_id':body.model_id,'status':'loaded'}
        return await start_worker(body.model_id)

@app.post('/api/unload')
async def unload():
    async with lock: await stop_worker(); return {'status':'unloaded'}

@app.get('/api/profiles')
def profiles():
    results=[]
    for p in PROFILES.glob('*.json'):
        try: results.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception: pass
    return results

@app.post('/api/profiles')
def add_profile(body:ProfileBody):
    try: data=base64.b64decode(body.audio_base64,validate=True)
    except Exception: raise HTTPException(400,'Invalid audio data')
    if len(data)>20_000_000: raise HTTPException(413,'Reference audio exceeds 20 MB')
    if not data: raise HTTPException(400,'Empty audio file')
    if data[:4]!=b'RIFF' or data[8:12]!=b'WAVE': raise HTTPException(400,'Reference must be a WAV file')
    id=uuid.uuid4().hex[:12]; audio=PROFILES/(id+'.wav'); audio.write_bytes(data)
    profile={'id':id,'name':body.name.strip() or 'Untitled voice','transcript':body.transcript.strip(),'path':str(audio)}
    (PROFILES/(id+'.json')).write_text(json.dumps(profile),encoding='utf-8')
    return profile

@app.post('/api/generate')
async def generate(body:GenerateBody):
    if not body.text.strip(): raise HTTPException(400,'Enter text first')
    if len(body.text)>8000: raise HTTPException(400,'Text is limited to 8,000 characters')
    if body.mode not in ('full','stream'): raise HTTPException(400,'Invalid generation mode')
    if body.model_id!=state['model']: raise HTTPException(409,'Select and load this model first')
    async def events():
        async with lock:
            proc=state['proc']
            if not proc or proc.returncode is not None:
                yield json.dumps({'type':'error','message':'Model worker is not running'})+'\n'; return
            payload=body.model_dump(); payload['type']='generate'; payload['output_dir']=str(GENERATED)
            if body.profile_id:
                p=PROFILES/(body.profile_id+'.json')
                if not p.exists(): yield json.dumps({'type':'error','message':'Saved voice not found'})+'\n'; return
                payload['profile']=json.loads(p.read_text(encoding='utf-8'))
            proc.stdin.write((json.dumps(payload)+'\n').encode()); await proc.stdin.drain()
            completed=False
            try:
                while True:
                    line=await proc.stdout.readline()
                    if not line:
                        yield json.dumps({'type':'error','message':'Worker stopped unexpectedly'})+'\n'; return
                    msg=json.loads(line)
                    if msg.get('file'):
                        msg['url']='/generated/'+Path(msg['file']).name
                        msg.pop('file',None)
                    yield json.dumps(msg)+'\n'
                    if msg.get('type') in ('done','error'):
                        completed=True
                        return
            except Exception as exc:
                yield json.dumps({'type':'error','message':f'Model stream failed: {exc}'})+'\n'
            finally:
                if not completed:
                    await asyncio.shield(stop_worker())
    return StreamingResponse(events(),media_type='application/x-ndjson',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.get('/generated/{filename}')
def audio(filename:str):
    if not re.fullmatch(r'[0-9a-f]{32}\.wav',filename): raise HTTPException(404)
    path=GENERATED/filename
    if not path.exists(): raise HTTPException(404)
    return FileResponse(path,media_type='audio/wav',filename=filename)

@app.get('/')
def root():
    path=ROOT/'dist/index.html'
    return FileResponse(path) if path.exists() else {'message':'Start the Vite frontend on port 5173'}

@app.get('/{path:path}')
def static(path:str):
    base=ROOT/'dist'; target=(base/path).resolve()
    if not str(target).startswith(str(base.resolve())) or not target.is_file(): raise HTTPException(404)
    return FileResponse(target)
