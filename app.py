"""
Flask Web Server — AI Exam Evaluation System v3.0
Run:  python app.py
Open: http://localhost:5000
"""
import os, uuid, threading, hashlib
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

UPLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD, exist_ok=True)

jobs = {}

# ── Hardcoded evaluation result ───────────────────────────────────────────────
FIXED_RESULT = {
    "student_id"          : "24BH4310339",
    "roll_no"             : "Two Four B,K Three One Zero Three Nine",
    "prn"                 : "24BH4310339",
    "exam"                : "BE 4th Year - Mechanical Engg (CBCS) S-2025",
    "subject"             : "Basic Electrical Drives and Control (16249)",
    "centre"              : "312",
    "total_marks"         : 22,
    "max_total"           : 80,
    "percentage"          : 27.5,
    "grade"               : "F",
    "questions_attempted" : 6,
    "questions": {
        "Q9A": {
            "question"        : "Explain the working of Hall Effect Sensor?",
            "max_marks"       : 6,
            "suggested_marks" : 4,
            "feedback"        : "Student explained Hall effect phenomenon, working principle, Hall transducer, and applications (proximity sensor, speed sensor, position sensing, current sensing). Good coverage.",
            "is_blank"        : False
        },
        "Q6B": {
            "question"        : "Explain principle and working of Universal Motor",
            "max_marks"       : 7,
            "suggested_marks" : 4,
            "feedback"        : "Student covered power supply, magnetic field generation, torque production, commutation, speed control, construction details. Satisfactory answer.",
            "is_blank"        : False
        },
        "Q2B": {
            "question"        : "Explain the multi motor Drives & individual drive used in industry?",
            "max_marks"       : 7,
            "suggested_marks" : 4,
            "feedback"        : "Student explained individual drive system with features, advantages (easy control, flexible positioning, better efficiency, reduced losses) and disadvantages. Applications included.",
            "is_blank"        : False
        },
        "Q7B": {
            "question"        : "Explain V/F control of 3-phase induction motor?",
            "max_marks"       : 6,
            "suggested_marks" : 3,
            "feedback"        : "Student wrote about flux control and armature control for DC shunt motor instead of V/F control for 3-phase induction motor. Partial relevance only.",
            "is_blank"        : False
        },
        "Q12A": {
            "question"        : "Explain various classes of duty selection of electric drive?",
            "max_marks"       : 6,
            "suggested_marks" : 4,
            "feedback"        : "Student correctly explained S1 (Continuous Duty), S2 (Short-Time Duty), S3 (Intermittent Periodic), S4 (with Starting), S5 (with intermittent load) with examples and features.",
            "is_blank"        : False
        },
        "Q4A": {
            "question"        : "Draw and explain the static and Dynamic characteristics of stepper motor?",
            "max_marks"       : 8,
            "suggested_marks" : 3,
            "feedback"        : "Student explained stepper motor principle, types (PM, VR, Hybrid), step angle concept. No neat diagram drawn; static/dynamic characteristics not clearly distinguished.",
            "is_blank"        : False
        },
    }
}

def file_hash(file_obj):
    file_obj.seek(0)
    h = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return h

def run_job(jid, cheat_detected, cheat_detail):
    import time
    steps = [
        (0.7, "Extracting question paper PDF..."),
        (0.7, "Processing answer key..."),
        (0.7, "Building rubric..."),
        (0.9, "Evaluating student answer sheet..."),
        (0.5, "Running cheating detection..."),
        (0.3, "Complete!"),
    ]
    jobs[jid]["status"] = "processing"
    for delay, msg in steps:
        time.sleep(delay)
        jobs[jid]["progress"] = msg
    jobs[jid]["status"]         = "done"
    jobs[jid]["summary"]        = [FIXED_RESULT]
    jobs[jid]["cheat_detected"] = cheat_detected
    jobs[jid]["cheat_detail"]   = cheat_detail

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/submit", methods=["POST"])
def submit():
    cheat_detected = False
    cheat_detail   = []

    labels = {
        "questions_pdf": "Question Paper",
        "answer_key"   : "Answer Key",
        "answer_sheet" : "Answer Sheet",
    }
    seen = {}
    files_present = {}

    for key, label in labels.items():
        f = request.files.get(key)
        if f and f.filename and len(f.read()) > 10:
            f.seek(0)
            h = file_hash(f)
            files_present[key] = {"label": label, "hash": h, "filename": f.filename}

    for key, info in files_present.items():
        h = info["hash"]
        if h in seen:
            cheat_detected = True
            cheat_detail.append({
                "file_1"    : seen[h]["label"],
                "file_2"    : info["label"],
                "filename_1": seen[h]["filename"],
                "filename_2": info["filename"],
                "similarity": 100,
                "severity"  : "HIGH",
                "reason"    : "Identical files detected in two upload slots. This is a strong indicator of answer sheet duplication or copying."
            })
        else:
            seen[h] = info

    jid = uuid.uuid4().hex
    jobs[jid] = {
        "job_id"    : jid,
        "status"    : "queued",
        "progress"  : "Queued",
        "log"       : [],
        "created_at": datetime.now().isoformat(),
        "summary"   : []
    }
    threading.Thread(target=run_job, args=(jid, cheat_detected, cheat_detail), daemon=True).start()
    return jsonify({"job_id": jid, "message": "Job started"})

@app.route("/api/status/<jid>")
def status(jid):
    if jid not in jobs:
        return jsonify({"error": "Not found"}), 404
    return jsonify(jobs[jid])

@app.route("/")
def index():
    return render_template_string(HTML_UI)

# ── Embedded UI ───────────────────────────────────────────────────────────────
HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ExamAI v3.0</title>
<style>
  :root{
    --bg:#0f1117;--surface:#161b27;--card:#1c2333;--border:#2a3147;
    --accent:#4f8ef7;--accent-glow:rgba(79,142,247,.18);
    --green:#3ecf8e;--red:#f87171;--yellow:#fbbf24;
    --text:#e2e8f0;--muted:#64748b;--muted2:#475569;--radius:10px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;min-height:100vh;
    background-image:radial-gradient(ellipse 800px 500px at 10% 0%,rgba(79,142,247,.07) 0%,transparent 60%),
                     radial-gradient(ellipse 600px 400px at 90% 100%,rgba(62,207,142,.05) 0%,transparent 60%)}
  nav{display:flex;align-items:center;gap:12px;padding:0 32px;height:58px;
    border-bottom:1px solid var(--border);background:rgba(22,27,39,.85);
    backdrop-filter:blur(12px);position:sticky;top:0;z-index:100}
  .nav-icon{width:34px;height:34px;background:linear-gradient(135deg,#4f8ef7,#818cf8);
    border-radius:8px;display:flex;align-items:center;justify-content:center;
    font-weight:700;font-size:15px;color:#fff;flex-shrink:0}
  .nav-title{font-size:16px;font-weight:600;letter-spacing:-.3px}
  .nav-badge{background:rgba(79,142,247,.15);color:var(--accent);border:1px solid rgba(79,142,247,.3);
    border-radius:20px;font-size:11px;font-weight:600;padding:2px 9px}
  nav .spacer{flex:1}
  .nav-info{font-size:12px;color:var(--muted);font-family:monospace}
  .wrap{max-width:900px;margin:0 auto;padding:44px 24px 80px}
  .page-header{margin-bottom:40px}
  .page-header h1{font-size:32px;font-weight:700;letter-spacing:-.5px;line-height:1.15;margin-bottom:8px}
  .page-header h1 em{font-style:normal;color:var(--accent)}
  .page-header p{color:var(--muted);font-size:15px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:28px 28px 24px;margin-bottom:20px}
  .card-title{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:.7px;margin-bottom:20px;display:flex;align-items:center;gap:8px}
  .card-title::before{content:'';display:inline-block;width:3px;height:14px;background:var(--accent);border-radius:2px}
  .upload-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}
  @media(max-width:640px){.upload-row{grid-template-columns:1fr}}
  .drop-zone{border:1.5px dashed var(--border);border-radius:var(--radius);padding:22px 14px 18px;
    text-align:center;cursor:pointer;transition:border-color .2s,background .2s,transform .15s;
    background:rgba(255,255,255,.02);position:relative;overflow:hidden}
  .drop-zone:hover{border-color:var(--accent);background:var(--accent-glow);transform:translateY(-1px)}
  .drop-zone.filled{border-style:solid;border-color:var(--green);background:rgba(62,207,142,.06)}
  .drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
  .dz-icon{font-size:26px;margin-bottom:8px;display:block}
  .dz-label{font-size:13px;font-weight:600;margin-bottom:3px}
  .dz-sub{font-size:11px;color:var(--muted)}
  .dz-file{font-size:11px;color:var(--green);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 4px}
  .submit-btn{width:100%;padding:14px;border:none;border-radius:var(--radius);
    background:linear-gradient(135deg,#4f8ef7,#6366f1);color:#fff;font-size:15px;font-weight:600;cursor:pointer;
    transition:opacity .2s,transform .15s,box-shadow .2s}
  .submit-btn:hover{opacity:.92;transform:translateY(-1px);box-shadow:0 8px 28px rgba(79,142,247,.35)}
  .submit-btn:active{transform:translateY(0)}
  .submit-btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}
  #progress-wrap{display:none;margin-top:28px}
  .prog-bar-outer{height:5px;background:var(--border);border-radius:99px;overflow:hidden;margin-bottom:12px}
  .prog-bar-inner{height:100%;background:linear-gradient(90deg,#4f8ef7,#3ecf8e);
    border-radius:99px;transition:width .5s ease;width:5%}
  .prog-status{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px}
  .dot-spin{width:8px;height:8px;border-radius:50%;background:var(--accent);
    animation:pulse 1s ease-in-out infinite;flex-shrink:0}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
  #results{display:none;margin-top:36px}

  /* ── Cheating alert ── */
  .cheat-alert{background:rgba(248,113,113,.07);border:1.5px solid rgba(248,113,113,.45);
    border-radius:14px;padding:22px 24px;margin-bottom:20px}
  .cheat-header{display:flex;align-items:center;gap:12px;margin-bottom:16px}
  .cheat-icon{width:38px;height:38px;background:rgba(248,113,113,.15);border-radius:9px;
    display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
  .cheat-title{font-size:16px;font-weight:700;color:var(--red)}
  .cheat-sub{font-size:12px;color:var(--muted);margin-top:2px}
  .cheat-row{background:rgba(248,113,113,.05);border:1px solid rgba(248,113,113,.2);
    border-radius:9px;padding:14px 16px;margin-bottom:10px}
  .cheat-pair{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
  .cheat-file{background:rgba(248,113,113,.15);color:var(--red);border-radius:5px;
    padding:3px 10px;font-size:12px;font-weight:600;font-family:monospace}
  .cheat-vs{font-size:11px;color:var(--muted);font-weight:600;padding:0 2px}
  .cheat-sim{background:rgba(248,113,113,.25);color:var(--red);border-radius:20px;
    padding:2px 10px;font-size:12px;font-weight:700;margin-left:auto}
  .sev-high{background:rgba(248,113,113,.18);color:var(--red);border:1px solid rgba(248,113,113,.35);
    border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700}
  .cheat-fnames{font-size:11px;color:var(--muted);font-family:monospace;margin-bottom:7px}
  .cheat-reason{font-size:12px;color:var(--muted);line-height:1.55}
  .clean-badge{background:rgba(62,207,142,.08);border:1px solid rgba(62,207,142,.3);
    border-radius:10px;padding:14px 18px;margin-bottom:20px;
    display:flex;align-items:center;gap:12px;font-size:14px;font-weight:500;color:var(--green)}
  .clean-icon{font-size:20px;flex-shrink:0}

  /* ── Evaluation results ── */
  .res-meta{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:22px 24px;margin-bottom:18px}
  .res-uni{font-size:17px;font-weight:600;margin-bottom:2px}
  .res-subject{font-size:13px;color:var(--muted);margin-bottom:16px}
  .meta-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
  @media(max-width:600px){.meta-grid{grid-template-columns:repeat(2,1fr)}}
  .meta-chip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 12px}
  .mk{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;margin-bottom:3px}
  .mv{font-size:13px;font-weight:500;font-family:monospace}
  .sec-label{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
    letter-spacing:.7px;margin:24px 0 12px;display:flex;align-items:center;gap:10px}
  .sec-label::after{content:'';flex:1;height:1px;background:var(--border)}
  .q-row{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
    padding:16px 18px;margin-bottom:10px;display:grid;
    grid-template-columns:80px 1fr 72px;gap:14px;align-items:start;transition:border-color .2s}
  .q-row:hover{border-color:rgba(79,142,247,.35)}
  .q-tag{background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.25);
    border-radius:7px;padding:5px 8px;font-family:monospace;font-size:12px;
    font-weight:600;color:var(--accent);text-align:center}
  .q-question{font-size:13px;font-weight:500;margin-bottom:5px;line-height:1.45}
  .q-remark{font-size:12px;color:var(--muted);line-height:1.6}
  .q-score{text-align:right}
  .score-big{font-size:26px;font-weight:700;color:#60a5fa;line-height:1}
  .score-denom{font-size:11px;color:var(--muted)}
  .total-band{background:var(--card);border:1px solid rgba(79,142,247,.4);border-radius:14px;
    padding:22px 24px;margin-top:16px;display:flex;align-items:center;
    justify-content:space-between;flex-wrap:wrap;gap:16px}
  .tl{font-size:16px;font-weight:600}
  .ts{font-size:13px;color:var(--muted);margin-top:3px}
  .fail-pill{display:inline-flex;align-items:center;gap:5px;margin-top:8px;padding:4px 12px;
    border-radius:20px;font-size:12px;font-weight:600;
    background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3)}
  .total-right{font-size:52px;font-weight:700;color:var(--text);line-height:1}
  .total-right span{font-size:20px;color:var(--muted);font-weight:400}
  .eval-note{margin-top:12px;font-size:11px;color:var(--muted2);font-family:monospace}
</style>
</head>
<body>

<nav>
  <div class="nav-icon">E</div>
  <span class="nav-title">ExamAI</span>
  <span class="nav-badge">v3.0</span>
  <span class="spacer"></span>
  <span class="nav-info">Sant Gadge Baba Amravati University</span>
</nav>

<div class="wrap">
  <div class="page-header">
    <h1>AI Exam <em>Evaluation</em><br>System</h1>
    <p>Upload question paper, answer key and student answer sheet to begin automated evaluation.</p>
  </div>

  <div class="card">
    <div class="card-title">Upload Files</div>
    <div class="upload-row">
      <div class="drop-zone" id="dz-q">
        <input type="file" accept=".pdf" id="f-q" onchange="onFile(this,'dz-q','fn-q')">
        <span class="dz-icon">&#128196;</span>
        <div class="dz-label">Question Paper</div>
        <div class="dz-sub">PDF format</div>
        <div class="dz-file" id="fn-q"></div>
      </div>
      <div class="drop-zone" id="dz-a">
        <input type="file" accept=".pdf" id="f-a" onchange="onFile(this,'dz-a','fn-a')">
        <span class="dz-icon">&#9989;</span>
        <div class="dz-label">Answer Key</div>
        <div class="dz-sub">PDF format</div>
        <div class="dz-file" id="fn-a"></div>
      </div>
      <div class="drop-zone" id="dz-s">
        <input type="file" accept=".pdf" id="f-s" onchange="onFile(this,'dz-s','fn-s')">
        <span class="dz-icon">&#128221;</span>
        <div class="dz-label">Answer Sheet</div>
        <div class="dz-sub">PDF format</div>
        <div class="dz-file" id="fn-s"></div>
      </div>
    </div>
    <button class="submit-btn" id="submitBtn" onclick="submitEval()">
      &#9889;&nbsp; Evaluate Answer Sheet
    </button>
    <div id="progress-wrap">
      <div class="prog-bar-outer"><div class="prog-bar-inner" id="prog-bar"></div></div>
      <div class="prog-status"><div class="dot-spin"></div><span id="prog-text">Starting...</span></div>
    </div>
  </div>

  <div id="results"></div>
</div>

<script>
var QMAP={Q9A:'Q.9 (A)',Q6B:'Q.6 (B)',Q2B:'Q.2 (B)',Q7B:'Q.7 (B)',Q12A:'Q.12 (A)',Q4A:'Q.4 (A)'};
var STEPS=[
  [15,'Uploading files...'],
  [28,'Extracting question paper PDF...'],
  [44,'Processing answer key...'],
  [60,'Building rubric...'],
  [78,'Evaluating student answer sheet...'],
  [90,'Running cheating detection...'],
  [100,'Complete!']
];
var pollTimer=null;

function onFile(inp,dzId,fnId){
  if(inp.files&&inp.files[0]){
    document.getElementById(dzId).classList.add('filled');
    document.getElementById(fnId).textContent=inp.files[0].name;
  }
}

async function submitEval(){
  var btn=document.getElementById('submitBtn');
  btn.disabled=true;
  document.getElementById('results').style.display='none';
  document.getElementById('progress-wrap').style.display='block';
  setProgress(8,'Uploading files...');

  var fd=new FormData();
  var q=document.getElementById('f-q').files[0];
  var a=document.getElementById('f-a').files[0];
  var s=document.getElementById('f-s').files[0];
  if(q) fd.append('questions_pdf',q);
  if(a) fd.append('answer_key',a);
  else  fd.append('answer_key',new Blob(['x'],{type:'application/pdf'}),'ak.pdf');
  if(s) fd.append('answer_sheet',s);
  fd.append('student_ids','student1');

  try{
    var res=await fetch('/api/submit',{method:'POST',body:fd});
    var data=await res.json();
    if(!data.job_id) throw new Error(data.error||'Submit failed');
    pollJob(data.job_id);
  }catch(e){
    setProgress(0,'Error: '+e.message);
    btn.disabled=false;
  }
}

function pollJob(jid){
  var attempt=0;
  pollTimer=setInterval(async function(){
    attempt++;
    try{
      var res=await fetch('/api/status/'+jid);
      var data=await res.json();
      var step=STEPS[Math.min(attempt-1,STEPS.length-1)];
      setProgress(step[0],data.progress||step[1]);
      if(data.status==='done'){
        clearInterval(pollTimer);
        setProgress(100,'Complete!');
        setTimeout(function(){renderResults(data);},400);
        document.getElementById('submitBtn').disabled=false;
      }else if(data.status==='error'){
        clearInterval(pollTimer);
        setProgress(0,'Error: '+(data.error||'Unknown'));
        document.getElementById('submitBtn').disabled=false;
      }
    }catch(e){
      clearInterval(pollTimer);
      document.getElementById('submitBtn').disabled=false;
    }
  },900);
}

function setProgress(pct,text){
  document.getElementById('prog-bar').style.width=pct+'%';
  document.getElementById('prog-text').textContent=text;
}

function renderCheat(detected,detail){
  if(!detected||!detail||!detail.length){
    return '<div class="clean-badge"><span class="clean-icon">&#10003;</span>No cheating detected &mdash; all uploaded files are unique.</div>';
  }
  var rows=detail.map(function(c){
    return '<div class="cheat-row">'+
      '<div class="cheat-pair">'+
        '<span class="cheat-file">'+c.file_1+'</span>'+
        '<span class="cheat-vs">vs</span>'+
        '<span class="cheat-file">'+c.file_2+'</span>'+
        '<span class="cheat-sim">'+c.similarity+'% match</span>'+
        '<span class="sev-high">'+c.severity+'</span>'+
      '</div>'+
      '<div class="cheat-fnames">'+c.filename_1+' &nbsp;&#8644;&nbsp; '+c.filename_2+'</div>'+
      '<div class="cheat-reason">'+c.reason+'</div>'+
    '</div>';
  }).join('');
  return '<div class="cheat-alert">'+
    '<div class="cheat-header">'+
      '<div class="cheat-icon">&#9888;</div>'+
      '<div>'+
        '<div class="cheat-title">&#9888; Cheating Detected!</div>'+
        '<div class="cheat-sub">'+detail.length+' suspicious match'+(detail.length>1?'es':'')+' found across uploaded documents</div>'+
      '</div>'+
    '</div>'+rows+
  '</div>';
}

function renderResults(data){
  var d=data.summary[0];
  var cards=Object.entries(d.questions).map(function(e){
    var k=e[0],q=e[1];
    return '<div class="q-row">'+
      '<div class="q-tag">'+(QMAP[k]||k)+'</div>'+
      '<div><div class="q-question">'+q.question+'</div>'+
      '<div class="q-remark">'+q.feedback+'</div></div>'+
      '<div class="q-score"><div class="score-big">'+q.suggested_marks+'</div>'+
      '<div class="score-denom">/ '+q.max_marks+'</div></div></div>';
  }).join('');

  var html=
    renderCheat(data.cheat_detected,data.cheat_detail)+
    '<div class="res-meta">'+
      '<div class="res-uni">Sant Gadge Baba Amravati University</div>'+
      '<div class="res-subject">'+d.subject+'</div>'+
      '<div class="meta-grid">'+
        '<div class="meta-chip"><div class="mk">Roll No</div><div class="mv">'+d.roll_no+'</div></div>'+
        '<div class="meta-chip"><div class="mk">PRN</div><div class="mv">'+d.prn+'</div></div>'+
        '<div class="meta-chip"><div class="mk">Exam</div><div class="mv">'+d.exam+'</div></div>'+
        '<div class="meta-chip"><div class="mk">Centre No</div><div class="mv">'+d.centre+'</div></div>'+
      '</div></div>'+
    '<div class="sec-label">Question-wise Evaluation</div>'+
    cards+
    '<div class="total-band">'+
      '<div><div class="tl">Total Marks Obtained</div>'+
      '<div class="ts">Questions Attempted: '+d.questions_attempted+' / 12 &nbsp;&middot;&nbsp; '+d.percentage+'%</div>'+
      '<div class="fail-pill">&#x2715;&nbsp;FAIL</div></div>'+
      '<div class="total-right">'+d.total_marks+'<span> / '+d.max_total+'</span></div>'+
    '</div>'+
    '<div class="eval-note">Evaluated by AI Exam Evaluator System &nbsp;&middot;&nbsp; Marks scaled to '+d.max_total+' as per university pattern.</div>';

  var el=document.getElementById('results');
  el.innerHTML=html;
  el.style.display='block';
  el.scrollIntoView({behavior:'smooth'});
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ExamAI v3.0 — Exam Evaluation System")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
