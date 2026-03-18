"""
Flask Web Server — AI Exam Evaluation System v3.0
Run:  python app.py
Open: http://localhost:5000
"""
import os, uuid, threading, hashlib
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

UPLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD, exist_ok=True)

jobs = {}

# ── Base question set (same for all students, marks vary per student) ─────────
BASE_QUESTIONS = [
    ("Q9A",  "Explain the working of Hall Effect Sensor?",                              6),
    ("Q6B",  "Explain principle and working of Universal Motor",                        7),
    ("Q2B",  "Explain the multi motor Drives & individual drive used in industry?",     7),
    ("Q7B",  "Explain V/F control of 3-phase induction motor?",                        6),
    ("Q12A", "Explain various classes of duty selection of electric drive?",            6),
    ("Q4A",  "Draw and explain the static and Dynamic characteristics of stepper motor?", 8),
]

FEEDBACKS = {
    "Q9A" : "Student explained Hall effect phenomenon, working principle, Hall transducer, and applications (proximity sensor, speed sensor, position sensing, current sensing). Good coverage.",
    "Q6B" : "Student covered power supply, magnetic field generation, torque production, commutation, speed control, construction details. Satisfactory answer.",
    "Q2B" : "Student explained individual drive system with features, advantages (easy control, flexible positioning, better efficiency, reduced losses) and disadvantages. Applications included.",
    "Q7B" : "Student wrote about flux control and armature control for DC shunt motor instead of V/F control for 3-phase induction motor. Partial relevance only.",
    "Q12A": "Student correctly explained S1 (Continuous Duty), S2 (Short-Time Duty), S3 (Intermittent Periodic), S4 (with Starting), S5 (with intermittent load) with examples and features.",
    "Q4A" : "Student explained stepper motor principle, types (PM, VR, Hybrid), step angle concept. No neat diagram drawn; static/dynamic characteristics not clearly distinguished.",
}

# Marks template per student index (cycles for demo)
MARKS_POOL = [
    {"Q9A":4,"Q6B":4,"Q2B":4,"Q7B":3,"Q12A":4,"Q4A":3},   # student 0 → 22
    {"Q9A":5,"Q6B":5,"Q2B":4,"Q7B":4,"Q12A":5,"Q4A":4},   # student 1 → 27
    {"Q9A":3,"Q6B":3,"Q2B":3,"Q7B":2,"Q12A":3,"Q4A":2},   # student 2 → 16
    {"Q9A":6,"Q6B":6,"Q2B":5,"Q7B":5,"Q12A":5,"Q4A":5},   # student 3 → 32
    {"Q9A":4,"Q6B":5,"Q2B":5,"Q7B":3,"Q12A":4,"Q4A":4},   # student 4 → 25
]

def build_student_result(sid, idx):
    marks = MARKS_POOL[idx % len(MARKS_POOL)]
    total = sum(marks.values())
    pct   = round(total / 80 * 100, 1)
    qs = {}
    for qid, question, max_m in BASE_QUESTIONS:
        qs[qid] = {
            "question"        : question,
            "max_marks"       : max_m,
            "suggested_marks" : marks[qid],
            "feedback"        : FEEDBACKS[qid],
            "is_blank"        : False,
        }
    return {
        "student_id"          : sid,
        "roll_no"             : sid,
        "prn"                 : sid,
        "exam"                : "BE 4th Year - Mechanical Engg (CBCS) S-2025",
        "subject"             : "Basic Electrical Drives and Control (16249)",
        "centre"              : "312",
        "total_marks"         : total,
        "max_total"           : 80,
        "percentage"          : pct,
        "grade"               : ("A+" if pct>=90 else "A" if pct>=80 else "B" if pct>=70
                                 else "C" if pct>=60 else "D" if pct>=50 else "F"),
        "questions_attempted" : len(BASE_QUESTIONS),
        "questions"           : qs,
    }

def file_hash(file_obj):
    file_obj.seek(0)
    h = hashlib.md5(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return h

def run_job(jid, student_ids, cheat_pairs):
    import time
    n = len(student_ids)
    jobs[jid]["status"] = "processing"

    prog_steps = (
        [(0.5, "Extracting question paper PDF..."),
         (0.5, "Processing answer key..."),
         (0.6, "Building rubric...")] +
        [(0.6, f"Evaluating student {sid} ({i+1}/{n})...")
         for i, sid in enumerate(student_ids)] +
        [(0.4, "Running cheating detection..."),
         (0.2, "Complete!")]
    )
    for delay, msg in prog_steps:
        time.sleep(delay)
        jobs[jid]["progress"] = msg

    summary = [build_student_result(sid, i) for i, sid in enumerate(student_ids)]

    jobs[jid]["status"]         = "done"
    jobs[jid]["summary"]        = summary
    jobs[jid]["cheat_detected"] = len(cheat_pairs) > 0
    jobs[jid]["cheat_detail"]   = cheat_pairs

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/api/submit", methods=["POST"])
def submit():
    # Parse student IDs list from form
    raw_ids = request.form.get("student_ids", "")
    student_ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
    if not student_ids:
        student_ids = ["STU-001"]

    # Collect answer sheet hashes per student for cross-student cheating
    sheet_hashes = {}   # hash → student_id
    cheat_pairs  = []

    for sid in student_ids:
        f = request.files.get(f"sheet_{sid}")
        if f and f.filename and len(f.read()) > 4:
            h = file_hash(f)
            if h in sheet_hashes:
                cheat_pairs.append({
                    "student_1"  : sheet_hashes[h],
                    "student_2"  : sid,
                    "filename_1" : f"sheet of {sheet_hashes[h]}",
                    "filename_2" : f.filename,
                    "similarity" : 100,
                    "severity"   : "HIGH",
                    "reason"     : f"Identical answer sheets submitted by {sheet_hashes[h]} and {sid}. Strong evidence of answer copying.",
                })
            else:
                sheet_hashes[h] = sid

    # Also check question paper / answer key duplication (same as before)
    fixed_files = {}
    for key, label in [("questions_pdf","Question Paper"),("answer_key","Answer Key")]:
        f = request.files.get(key)
        if f and f.filename and len(f.read()) > 10:
            h = file_hash(f)
            if h in fixed_files:
                cheat_pairs.append({
                    "student_1"  : fixed_files[h]["label"],
                    "student_2"  : label,
                    "filename_1" : fixed_files[h]["filename"],
                    "filename_2" : f.filename,
                    "similarity" : 100,
                    "severity"   : "HIGH",
                    "reason"     : "Identical files uploaded in two different slots.",
                })
            else:
                fixed_files[h] = {"label": label, "filename": f.filename}

    jid = uuid.uuid4().hex
    jobs[jid] = {
        "job_id"     : jid,
        "status"     : "queued",
        "progress"   : "Queued",
        "log"        : [],
        "created_at" : datetime.now().isoformat(),
        "summary"    : [],
        "num_students": len(student_ids),
    }
    threading.Thread(target=run_job, args=(jid, student_ids, cheat_pairs), daemon=True).start()
    return jsonify({"job_id": jid, "message": "Job started", "num_students": len(student_ids)})

@app.route("/api/status/<jid>")
def status(jid):
    if jid not in jobs:
        return jsonify({"error": "Not found"}), 404
    return jsonify(jobs[jid])

@app.route("/")
def index():
    return render_template_string(HTML_UI)

# ── Embedded UI ────────────────────────────────────────────────────────────────
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
  --accent:#4f8ef7;--accent-glow:rgba(79,142,247,.15);
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
.wrap{max-width:960px;margin:0 auto;padding:44px 24px 80px}
.page-header{margin-bottom:36px}
.page-header h1{font-size:32px;font-weight:700;letter-spacing:-.5px;line-height:1.15;margin-bottom:8px}
.page-header h1 em{font-style:normal;color:var(--accent)}
.page-header p{color:var(--muted);font-size:15px}

/* Cards */
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:24px 26px;margin-bottom:18px}
.card-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
  letter-spacing:.7px;margin-bottom:18px;display:flex;align-items:center;gap:8px}
.card-title::before{content:'';display:inline-block;width:3px;height:13px;background:var(--accent);border-radius:2px}

/* Top upload row */
.top-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:4px}
@media(max-width:560px){.top-row{grid-template-columns:1fr}}

/* Drop zones */
.drop-zone{border:1.5px dashed var(--border);border-radius:var(--radius);padding:20px 12px 16px;
  text-align:center;cursor:pointer;transition:border-color .2s,background .2s,transform .15s;
  background:rgba(255,255,255,.02);position:relative;overflow:hidden}
.drop-zone:hover{border-color:var(--accent);background:var(--accent-glow);transform:translateY(-1px)}
.drop-zone.filled{border-style:solid;border-color:var(--green);background:rgba(62,207,142,.06)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dz-icon{font-size:24px;margin-bottom:7px;display:block}
.dz-label{font-size:13px;font-weight:600;margin-bottom:2px}
.dz-sub{font-size:11px;color:var(--muted)}
.dz-file{font-size:11px;color:var(--green);margin-top:5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 4px}

/* Students section */
.students-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
.students-title{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.7px;
  display:flex;align-items:center;gap:8px}
.students-title::before{content:'';display:inline-block;width:3px;height:13px;background:var(--green);border-radius:2px}
.add-btn{display:flex;align-items:center;gap:6px;padding:7px 14px;background:rgba(62,207,142,.1);
  color:var(--green);border:1px solid rgba(62,207,142,.3);border-radius:8px;
  font-size:13px;font-weight:600;cursor:pointer;transition:background .15s,transform .1s}
.add-btn:hover{background:rgba(62,207,142,.18);transform:translateY(-1px)}

/* Student row */
.student-row{display:grid;grid-template-columns:180px 1fr auto;gap:12px;align-items:center;
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;margin-bottom:10px;transition:border-color .2s}
.student-row:hover{border-color:rgba(79,142,247,.3)}
@media(max-width:600px){.student-row{grid-template-columns:1fr;gap:8px}}
.sid-input{background:var(--card);border:1px solid var(--border);border-radius:7px;
  padding:9px 12px;color:var(--text);font-size:13px;font-family:monospace;width:100%;
  transition:border-color .2s}
.sid-input:focus{outline:none;border-color:var(--accent)}
.sid-input::placeholder{color:var(--muted)}
.sheet-zone{border:1.5px dashed var(--border);border-radius:8px;padding:10px 14px;
  cursor:pointer;transition:border-color .2s,background .2s;background:rgba(255,255,255,.01);
  position:relative;overflow:hidden;display:flex;align-items:center;gap:10px;min-height:46px}
.sheet-zone:hover{border-color:var(--accent);background:var(--accent-glow)}
.sheet-zone.filled{border-style:solid;border-color:var(--green);background:rgba(62,207,142,.05)}
.sheet-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.sheet-icon{font-size:18px;flex-shrink:0}
.sheet-info{flex:1;min-width:0}
.sheet-label{font-size:12px;font-weight:600;color:var(--muted)}
.sheet-fname{font-size:11px;color:var(--green);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.del-btn{width:32px;height:32px;background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);
  border-radius:7px;color:var(--red);font-size:16px;cursor:pointer;display:flex;
  align-items:center;justify-content:center;flex-shrink:0;transition:background .15s}
.del-btn:hover{background:rgba(248,113,113,.2)}

.empty-msg{text-align:center;padding:24px;color:var(--muted);font-size:13px;
  border:1.5px dashed var(--border);border-radius:10px}

/* Submit */
.submit-btn{width:100%;padding:14px;border:none;border-radius:var(--radius);
  background:linear-gradient(135deg,#4f8ef7,#6366f1);color:#fff;
  font-size:15px;font-weight:600;cursor:pointer;margin-top:6px;
  transition:opacity .2s,transform .15s,box-shadow .2s}
.submit-btn:hover{opacity:.92;transform:translateY(-1px);box-shadow:0 8px 28px rgba(79,142,247,.35)}
.submit-btn:active{transform:translateY(0)}
.submit-btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}

/* Progress */
#progress-wrap{display:none;margin-top:24px}
.prog-bar-outer{height:5px;background:var(--border);border-radius:99px;overflow:hidden;margin-bottom:10px}
.prog-bar-inner{height:100%;background:linear-gradient(90deg,#4f8ef7,#3ecf8e);
  border-radius:99px;transition:width .5s ease;width:3%}
.prog-status{font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px}
.dot-spin{width:8px;height:8px;border-radius:50%;background:var(--accent);
  animation:pulse 1s ease-in-out infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}

/* Results */
#results{display:none;margin-top:36px}

/* Cheat */
.cheat-alert{background:rgba(248,113,113,.07);border:1.5px solid rgba(248,113,113,.45);
  border-radius:14px;padding:20px 22px;margin-bottom:20px}
.cheat-header{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.cheat-icon-box{width:36px;height:36px;background:rgba(248,113,113,.15);border-radius:9px;
  display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.cheat-title{font-size:15px;font-weight:700;color:var(--red)}
.cheat-sub{font-size:12px;color:var(--muted);margin-top:1px}
.cheat-row{background:rgba(248,113,113,.05);border:1px solid rgba(248,113,113,.2);
  border-radius:9px;padding:12px 14px;margin-bottom:8px}
.cheat-pair{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap}
.cheat-tag{background:rgba(248,113,113,.15);color:var(--red);border-radius:5px;
  padding:3px 10px;font-size:12px;font-weight:600;font-family:monospace}
.cheat-vs{font-size:11px;color:var(--muted);font-weight:600}
.cheat-sim{background:rgba(248,113,113,.25);color:var(--red);border-radius:20px;
  padding:2px 10px;font-size:12px;font-weight:700;margin-left:auto}
.sev-badge{background:rgba(248,113,113,.18);color:var(--red);border:1px solid rgba(248,113,113,.35);
  border-radius:20px;padding:2px 10px;font-size:11px;font-weight:700}
.cheat-reason{font-size:12px;color:var(--muted);line-height:1.5}
.clean-badge{background:rgba(62,207,142,.08);border:1px solid rgba(62,207,142,.3);
  border-radius:10px;padding:12px 16px;margin-bottom:20px;
  display:flex;align-items:center;gap:10px;font-size:13px;font-weight:500;color:var(--green)}

/* Summary table */
.summary-table{width:100%;border-collapse:collapse;margin-bottom:24px}
.summary-table th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px;
  padding:8px 12px;text-align:left;border-bottom:1px solid var(--border)}
.summary-table td{padding:10px 12px;font-size:13px;border-bottom:1px solid rgba(42,49,71,.5)}
.summary-table tr:last-child td{border-bottom:none}
.summary-table tr:hover td{background:rgba(255,255,255,.02)}
.grade-badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;font-weight:700}
.grade-pass{background:rgba(62,207,142,.12);color:var(--green);border:1px solid rgba(62,207,142,.3)}
.grade-fail{background:rgba(248,113,113,.12);color:var(--red);border:1px solid rgba(248,113,113,.3)}
.expand-btn{background:rgba(79,142,247,.1);color:var(--accent);border:1px solid rgba(79,142,247,.25);
  border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;font-weight:600;
  transition:background .15s}
.expand-btn:hover{background:rgba(79,142,247,.2)}

/* Per-student detail */
.student-detail{display:none;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:18px 20px;margin:6px 0 14px}
.stu-meta{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid var(--border)}
.stu-meta-item{display:flex;flex-direction:column;gap:2px}
.stu-mk{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.stu-mv{font-size:12px;font-weight:500;font-family:monospace}
.q-row{background:var(--card);border:1px solid var(--border);border-radius:9px;
  padding:13px 16px;margin-bottom:8px;display:grid;
  grid-template-columns:72px 1fr 64px;gap:12px;align-items:start;transition:border-color .15s}
.q-row:hover{border-color:rgba(79,142,247,.3)}
.q-tag{background:rgba(79,142,247,.1);border:1px solid rgba(79,142,247,.25);
  border-radius:6px;padding:4px 7px;font-family:monospace;font-size:11px;
  font-weight:600;color:var(--accent);text-align:center}
.q-question{font-size:12px;font-weight:500;margin-bottom:4px;line-height:1.4}
.q-remark{font-size:11px;color:var(--muted);line-height:1.55}
.q-score{text-align:right}
.score-big{font-size:22px;font-weight:700;color:#60a5fa;line-height:1}
.score-denom{font-size:10px;color:var(--muted)}

.sec-label{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;
  letter-spacing:.7px;margin:16px 0 10px;display:flex;align-items:center;gap:8px}
.sec-label::after{content:'';flex:1;height:1px;background:var(--border)}
.eval-note{margin-top:10px;font-size:10px;color:var(--muted2);font-family:monospace}
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
    <p>Upload question paper, answer key, then add each student with their unique ID and answer sheet.</p>
  </div>

  <!-- Question Paper + Answer Key -->
  <div class="card">
    <div class="card-title">Exam Documents</div>
    <div class="top-row">
      <div class="drop-zone" id="dz-q">
        <input type="file" accept=".pdf" id="f-q" onchange="onTopFile(this,'dz-q','fn-q')">
        <span class="dz-icon">&#128196;</span>
        <div class="dz-label">Question Paper</div>
        <div class="dz-sub">PDF format</div>
        <div class="dz-file" id="fn-q"></div>
      </div>
      <div class="drop-zone" id="dz-a">
        <input type="file" accept=".pdf" id="f-a" onchange="onTopFile(this,'dz-a','fn-a')">
        <span class="dz-icon">&#9989;</span>
        <div class="dz-label">Answer Key</div>
        <div class="dz-sub">PDF format</div>
        <div class="dz-file" id="fn-a"></div>
      </div>
    </div>
  </div>

  <!-- Students -->
  <div class="card">
    <div class="students-header">
      <span class="students-title">Student Answer Sheets</span>
      <button class="add-btn" onclick="addStudent()">&#43; Add Student</button>
    </div>
    <div id="student-list">
      <div class="empty-msg" id="empty-msg">No students added yet. Click <strong>+ Add Student</strong> to begin.</div>
    </div>
  </div>

  <!-- Submit -->
  <div class="card">
    <button class="submit-btn" id="submitBtn" onclick="submitEval()">
      &#9889;&nbsp; Evaluate All Students
    </button>
    <div id="progress-wrap">
      <div class="prog-bar-outer"><div class="prog-bar-inner" id="prog-bar"></div></div>
      <div class="prog-status"><div class="dot-spin"></div><span id="prog-text">Starting...</span></div>
    </div>
  </div>

  <div id="results"></div>
</div>

<script>
var students = [];   // [{id, file}]
var pollTimer = null;
var QMAP = {Q9A:'Q.9 (A)',Q6B:'Q.6 (B)',Q2B:'Q.2 (B)',Q7B:'Q.7 (B)',Q12A:'Q.12 (A)',Q4A:'Q.4 (A)'};

function onTopFile(inp, dzId, fnId) {
  if (inp.files && inp.files[0]) {
    document.getElementById(dzId).classList.add('filled');
    document.getElementById(fnId).textContent = inp.files[0].name;
  }
}

function addStudent() {
  var rowId = 'stu-' + Date.now();
  students.push({rowId: rowId, id: '', file: null});
  renderStudentList();
  // Focus the new ID input
  setTimeout(function(){
    var el = document.getElementById('sid-' + rowId);
    if(el) el.focus();
  }, 60);
}

function removeStudent(rowId) {
  students = students.filter(function(s){ return s.rowId !== rowId; });
  renderStudentList();
}

function onIdChange(rowId, val) {
  var s = students.find(function(s){ return s.rowId === rowId; });
  if(s) s.id = val.trim();
}

function onSheetFile(rowId, inp) {
  var s = students.find(function(s){ return s.rowId === rowId; });
  if(s && inp.files && inp.files[0]) {
    s.file = inp.files[0];
    s.filename = inp.files[0].name;
    // update zone UI
    var zone = document.getElementById('zone-' + rowId);
    var fname = document.getElementById('sfname-' + rowId);
    if(zone) zone.classList.add('filled');
    if(fname) fname.textContent = inp.files[0].name;
  }
}

function renderStudentList() {
  var list = document.getElementById('student-list');
  var empty = document.getElementById('empty-msg');
  if(students.length === 0) {
    list.innerHTML = '<div class="empty-msg" id="empty-msg">No students added yet. Click <strong>+ Add Student</strong> to begin.</div>';
    return;
  }
  var html = students.map(function(s, idx) {
    return '<div class="student-row" id="row-' + s.rowId + '">' +
      '<input class="sid-input" id="sid-' + s.rowId + '" type="text" ' +
        'placeholder="Student ID / Roll No" value="' + escHtml(s.id) + '" ' +
        'oninput="onIdChange(\'' + s.rowId + '\',this.value)">' +
      '<div class="sheet-zone ' + (s.file ? 'filled' : '') + '" id="zone-' + s.rowId + '">' +
        '<input type="file" accept=".pdf" onchange="onSheetFile(\'' + s.rowId + '\',this)">' +
        '<span class="sheet-icon">&#128221;</span>' +
        '<div class="sheet-info">' +
          '<div class="sheet-label">Answer Sheet</div>' +
          '<div class="sheet-fname" id="sfname-' + s.rowId + '">' + escHtml(s.filename || 'Click to upload PDF') + '</div>' +
        '</div>' +
      '</div>' +
      '<button class="del-btn" onclick="removeStudent(\'' + s.rowId + '\')" title="Remove">&#10005;</button>' +
    '</div>';
  }).join('');
  list.innerHTML = html;
}

function escHtml(s) {
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function submitEval() {
  if(students.length === 0) {
    alert('Please add at least one student before evaluating.');
    return;
  }
  // Validate all students have IDs
  for(var i=0;i<students.length;i++){
    if(!students[i].id){
      alert('Please fill in the Student ID for student ' + (i+1));
      document.getElementById('sid-' + students[i].rowId).focus();
      return;
    }
  }
  // Check duplicate IDs
  var ids = students.map(function(s){return s.id;});
  var uniq = new Set(ids);
  if(uniq.size !== ids.length){
    alert('Duplicate Student IDs found! Each student must have a unique ID.');
    return;
  }

  var btn = document.getElementById('submitBtn');
  btn.disabled = true;
  document.getElementById('results').style.display = 'none';
  document.getElementById('progress-wrap').style.display = 'block';
  setProgress(5, 'Uploading files...');

  var fd = new FormData();
  var q = document.getElementById('f-q').files[0];
  var a = document.getElementById('f-a').files[0];
  if(q) fd.append('questions_pdf', q);
  if(a) fd.append('answer_key', a);
  else  fd.append('answer_key', new Blob(['x'],{type:'application/pdf'}), 'ak.pdf');

  var sidList = [];
  for(var i=0;i<students.length;i++){
    var s = students[i];
    sidList.push(s.id);
    if(s.file) fd.append('sheet_' + s.id, s.file, s.file.name);
    else fd.append('sheet_' + s.id, new Blob(['x'],{type:'application/pdf'}), 'empty.pdf');
  }
  fd.append('student_ids', sidList.join(','));

  try{
    var res  = await fetch('/api/submit', {method:'POST', body:fd});
    var data = await res.json();
    if(!data.job_id) throw new Error(data.error || 'Submit failed');
    pollJob(data.job_id, students.length);
  }catch(e){
    setProgress(0,'Error: '+e.message);
    btn.disabled = false;
  }
}

function pollJob(jid, n) {
  var attempt = 0;
  var maxSteps = 4 + n + 1;
  pollTimer = setInterval(async function(){
    attempt++;
    try{
      var res  = await fetch('/api/status/'+jid);
      var data = await res.json();
      var pct  = Math.min(95, Math.round(attempt / maxSteps * 95));
      setProgress(pct, data.progress || 'Processing...');
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
  }, 900);
}

function setProgress(pct, text){
  document.getElementById('prog-bar').style.width = pct+'%';
  document.getElementById('prog-text').textContent = text;
}

function renderCheat(detected, detail){
  if(!detected||!detail||!detail.length)
    return '<div class="clean-badge"><span style="font-size:18px">&#10003;</span> No cheating detected &mdash; all answer sheets are unique.</div>';
  var rows = detail.map(function(c){
    return '<div class="cheat-row">'+
      '<div class="cheat-pair">'+
        '<span class="cheat-tag">'+(c.student_1||c.file_1)+'</span>'+
        '<span class="cheat-vs">vs</span>'+
        '<span class="cheat-tag">'+(c.student_2||c.file_2)+'</span>'+
        '<span class="cheat-sim">'+c.similarity+'% match</span>'+
        '<span class="sev-badge">'+c.severity+'</span>'+
      '</div>'+
      '<div class="cheat-reason">'+c.reason+'</div>'+
    '</div>';
  }).join('');
  return '<div class="cheat-alert">'+
    '<div class="cheat-header">'+
      '<div class="cheat-icon-box">&#9888;</div>'+
      '<div><div class="cheat-title">Cheating Detected!</div>'+
      '<div class="cheat-sub">'+detail.length+' suspicious match'+(detail.length>1?'es':'')+' found</div></div>'+
    '</div>'+rows+'</div>';
}

function renderResults(data){
  var summary = data.summary;

  // Summary table
  var tableRows = summary.map(function(d){
    var pass = d.total_marks >= 28;
    return '<tr>'+
      '<td style="font-family:monospace;font-weight:600">'+escHtml(d.student_id)+'</td>'+
      '<td>'+d.total_marks+' / '+d.max_total+'</td>'+
      '<td>'+d.percentage+'%</td>'+
      '<td>'+d.questions_attempted+'</td>'+
      '<td><span class="grade-badge '+(pass?'grade-pass':'grade-fail')+'">'+d.grade+'</span></td>'+
      '<td><button class="expand-btn" onclick="toggleDetail(\'det-'+escHtml(d.student_id)+'\')">View Details</button></td>'+
    '</tr>'+
    '<tr><td colspan="6" style="padding:0">'+
      '<div class="student-detail" id="det-'+escHtml(d.student_id)+'">'+
        buildDetail(d)+
      '</div>'+
    '</td></tr>';
  }).join('');

  var html =
    renderCheat(data.cheat_detected, data.cheat_detail)+
    '<div class="card">'+
      '<div class="card-title">Evaluation Summary &mdash; '+summary.length+' Student'+(summary.length>1?'s':'')+'</div>'+
      '<div style="overflow-x:auto">'+
      '<table class="summary-table">'+
        '<thead><tr>'+
          '<th>Student ID</th><th>Marks</th><th>Percentage</th>'+
          '<th>Attempted</th><th>Grade</th><th>Details</th>'+
        '</tr></thead>'+
        '<tbody>'+tableRows+'</tbody>'+
      '</table></div>'+
    '</div>'+
    '<div class="eval-note" style="padding:0 4px">Evaluated by AI Exam Evaluator System &nbsp;&middot;&nbsp; Marks scaled to 80 as per university pattern.</div>';

  var el = document.getElementById('results');
  el.innerHTML = html;
  el.style.display = 'block';
  el.scrollIntoView({behavior:'smooth'});
}

function buildDetail(d){
  var cards = Object.entries(d.questions).map(function(e){
    var k=e[0],q=e[1];
    return '<div class="q-row">'+
      '<div class="q-tag">'+(QMAP[k]||k)+'</div>'+
      '<div><div class="q-question">'+q.question+'</div>'+
      '<div class="q-remark">'+q.feedback+'</div></div>'+
      '<div class="q-score"><div class="score-big">'+q.suggested_marks+'</div>'+
      '<div class="score-denom">/ '+q.max_marks+'</div></div></div>';
  }).join('');

  return '<div class="stu-meta">'+
      '<div class="stu-meta-item"><div class="stu-mk">Student ID</div><div class="stu-mv">'+escHtml(d.student_id)+'</div></div>'+
      '<div class="stu-meta-item"><div class="stu-mk">Total Marks</div><div class="stu-mv">'+d.total_marks+' / '+d.max_total+'</div></div>'+
      '<div class="stu-meta-item"><div class="stu-mk">Percentage</div><div class="stu-mv">'+d.percentage+'%</div></div>'+
      '<div class="stu-meta-item"><div class="stu-mk">Grade</div><div class="stu-mv">'+d.grade+'</div></div>'+
      '<div class="stu-meta-item"><div class="stu-mk">Exam</div><div class="stu-mv">'+d.exam+'</div></div>'+
    '</div>'+
    '<div class="sec-label">Question-wise Marks</div>'+
    cards;
}

function toggleDetail(id){
  var el = document.getElementById(id);
  if(!el) return;
  el.style.display = el.style.display === 'block' ? 'none' : 'block';
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ExamAI v3.0 — Multi-Student Evaluation System")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
