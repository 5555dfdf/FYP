"""Read-only audit of source IRDS data; derived outputs stay in analysis/."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis'
SIM = ROOT / 'SkeletonData' / 'Simplified'
RAW = ROOT / 'SkeletonData' / 'RawData'
NAMES = ['Left elbow flexion', 'Right elbow flexion', 'Left shoulder flexion',
         'Right shoulder flexion', 'Left shoulder abduction', 'Right shoulder abduction',
         'Both arms elevation', 'Left side tap', 'Right side tap']
JOINTS = ['SpineBase','SpineMid','Neck','Head','ShoulderLeft','ElbowLeft','WristLeft',
          'HandLeft','ShoulderRight','ElbowRight','WristRight','HandRight','HipLeft',
          'KneeLeft','AnkleLeft','FootLeft','HipRight','KneeRight','AnkleRight','FootRight',
          'SpineShoulder','HandTipLeft','ThumbLeft','HandTipRight','ThumbRight']
EDGES = [(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),
         (20,8),(8,9),(9,10),(10,11),(0,12),(12,13),(13,14),(14,15),
         (0,16),(16,17),(17,18),(18,19)]

def count_by(rows, keys):
    c = Counter(tuple(r[k] for k in keys) for r in rows)
    return [{'|'.join(keys): '|'.join(map(str,k)), 'count': v} for k,v in sorted(c.items())]

def angle(a,b):
    d = np.linalg.norm(a,axis=-1) * np.linalg.norm(b,axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b,axis=-1)/np.maximum(d,1e-10),-1,1)))

rows, arrays, failures = [], {}, []
raw_states, joint_states = Counter(), {j:Counter() for j in JOINTS}
hashes = defaultdict(list)
raw_mismatch, dt_hist, frame_gaps = [], Counter(), Counter()
extra_headers=[]
joint_pattern = re.compile(r'\(([^()]*)\)')
for i,p in enumerate(sorted(SIM.glob('*.txt'))):
    subject,date,gesture,rep,label,position = p.stem.split('_',5)
    row = dict(file=p.name, subject=int(subject), date=int(date), gesture=int(gesture),
               repetition=int(rep), label=int(label), position=position,
               group='patient' if subject.startswith('2') else 'control')
    try:
        x = np.loadtxt(p, delimiter=',',ndmin=2)
        if x.shape[1]!=75 or not np.isfinite(x).all():
            raise ValueError(f'invalid shape or values: {x.shape}')
        row.update(frames=len(x), seconds_nominal=len(x)/30)
        arrays[p.name] = x.reshape(-1,25,3)
        hashes[hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()].append(p.name)
        rp = RAW/p.name
        lines = rp.read_text(encoding='utf-8-sig').splitlines()
        if lines[0] != 'Version0.1':
            raise ValueError(f'unknown raw header {lines[0]}')
        raw_xyz, stamps, frame_ids = [], [], []
        seq_states = Counter()
        headers=sum(line=='Version0.1' for line in lines)
        row['raw_version_headers']=headers
        if headers>1: extra_headers.append({'file':p.name,'headers':headers})
        for line in lines[1:]:
            if line=='Version0.1': continue
            fields = line.split(',',3)
            frame_ids.append(int(fields[0])); stamps.append(float(fields[2]))
            joints = [g.split(',') for g in joint_pattern.findall(line)]
            if [g[0] for g in joints] != JOINTS:
                raise ValueError('unexpected joint order')
            raw_xyz.append([float(v) for g in joints for v in g[2:5]])
            for g in joints:
                raw_states[g[1]] += 1; joint_states[g[0]][g[1]] += 1; seq_states[g[1]] += 1
        raw_array = np.array(raw_xyz)
        match = raw_array.shape==x.shape and np.array_equal(raw_array,x)
        if not match:
            raw_mismatch.append(p.name)
        row['raw_frames'] = len(raw_xyz)
        row['raw_xyz_exact_match'] = match
        row['tracking_states'] = dict(seq_states)
        row['timestamp_span_raw_units'] = stamps[-1]-stamps[0]
        row['timestamp_nonpositive_deltas'] = int(np.sum(np.diff(stamps)<=0))
        row['timestamp_max_delta_raw_units'] = float(np.max(np.diff(stamps)))
        row['frame_id_nonunit_deltas'] = int(np.sum(np.diff(frame_ids)!=1))
        dt_hist.update(np.diff(stamps).tolist())
        frame_gaps.update(np.diff(frame_ids).tolist())
    except Exception as e:
        failures.append({'file':p.name,'error':str(e)})
    rows.append(row)
    if (i+1)%500==0:
        print(f'Audited {i+1} paired sequences',flush=True)

valid = [r for r in rows if r['label'] in [1,2] and 'frames' in r]
def lens(rs):
    v = np.array([r['frames'] for r in rs])
    return dict(n=len(v),min=int(v.min()),max=int(v.max()),mean=float(v.mean()),
                median=float(np.median(v)),p75=float(np.percentile(v,75)),sum_frames=int(v.sum()))

summary = dict(
    sequences=len(rows), subjects=sorted(set(r['subject'] for r in rows)),
    subject_counts_by_group={g:len(set(r['subject'] for r in rows if r['group']==g)) for g in ['control','patient']},
    date_ids=sorted(set(r['date'] for r in rows)), label_counts=count_by(rows,['label']),
    gesture_label=count_by(rows,['gesture','label']), group_label=count_by(rows,['group','label']),
    subject_label=count_by(rows,['subject','label']), valid_position=count_by(valid,['position']),
    subject_position_label=count_by(valid,['subject','position','label']),
    gesture_subject_counts={str(g):{str(l):len(set(r['subject'] for r in valid if r['gesture']==g and r['label']==l)) for l in [1,2]} for g in range(9)},
    lengths_all=lens(rows), lengths_binary=lens(valid),
    lengths_by_label={str(l):lens([r for r in rows if r['label']==l]) for l in [1,2,3]},
    tracking_states=dict(raw_states), joint_tracking_states={j:dict(c) for j,c in joint_states.items()},
    raw_xyz_mismatches=raw_mismatch, raw_timestamp_delta_counts=dict(dt_hist.most_common(12)),
    raw_frame_id_delta_counts=dict(frame_gaps.most_common(12)),
    source_pair_names_equal={p.name for p in SIM.glob('*.txt')}=={p.name for p in RAW.glob('*.txt')},
    raw_repeated_version_headers=extra_headers,
    sequences_with_nonpositive_timestamp_delta=sum(r.get('timestamp_nonpositive_deltas',0)>0 for r in rows),
    sequences_with_timestamp_gap_over_1000_units=sum(r.get('timestamp_max_delta_raw_units',0)>1000 for r in rows),
    sequences_with_nonunit_frame_id_step=sum(r.get('frame_id_nonunit_deltas',0)>0 for r in rows),
    duplicate_coordinate_sequences=[v for v in hashes.values() if len(v)>1], failures=failures)
side_checks=[]
for g in [2,3,4,5]:
    rs=[r for r in valid if r['gesture']==g and r['label']==1 and r['group']=='control']
    peaks=[]
    for r in rs:
        x=arrays[r['file']]
        left=angle(x[:,5]-x[:,4],x[:,0]-x[:,20])
        right=angle(x[:,9]-x[:,8],x[:,0]-x[:,20])
        peaks.append((float(left.max()),float(right.max())))
    side_checks.append(dict(gesture=g,correct_control_sequences=len(rs),
                            raw_left_peak_larger=sum(l>r for l,r in peaks),
                            raw_right_peak_larger=sum(r>l for l,r in peaks),
                            median_raw_left_peak=float(np.median([l for l,r in peaks])),
                            median_raw_right_peak=float(np.median([r for l,r in peaks]))))
summary['bilateral_shoulder_check']=side_checks
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'sample_manifest.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs = plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
colors={1:'#287f79',2:'#c66a40',3:'#858898'}
for l in [1,2,3]:
    axs[0,0].bar(np.arange(9)+(l-2)*.25,[sum(r['gesture']==g and r['label']==l for r in rows) for g in range(9)],width=.25,color=colors[l],label={1:'Correct (1)',2:'Incorrect (2)',3:'Unassignable / poor (3)'}[l])
axs[0,0].set(xticks=np.arange(9),xlabel='Gesture ID',ylabel='Sequences',title='All 2,589 sequences: gesture and label')
axs[0,0].legend(fontsize=8)
subs=summary['subjects']
bottom=np.zeros(len(subs))
for l in [1,2,3]:
    heights=np.array([sum(r['subject']==s and r['label']==l for r in rows) for s in subs])
    axs[0,1].bar(np.arange(len(subs)),heights,bottom=bottom,color=colors[l]); bottom+=heights
axs[0,1].set(xticks=np.arange(len(subs)),xticklabels=subs,ylabel='Sequences',title='Subject distribution (2xx = patients)')
axs[0,1].tick_params(axis='x',rotation=90,labelsize=8)
for l in [1,2]:
    axs[1,0].hist([r['frames']/30 for r in valid if r['label']==l],bins=np.linspace(0,54,55),alpha=.65,color=colors[l],label=f'Label {l}')
axs[1,0].set(xlabel='Nominal duration (frames / 30), seconds',ylabel='Sequences',title='Duration is associated with correctness')
axs[1,0].legend()
positions=sorted(set(r['position'] for r in valid))
for i,g in enumerate(['control','patient']):
    vals=[sum(r['position']==p and r['group']==g for r in valid) for p in positions]
    bars=axs[1,1].bar(np.arange(len(positions))+(i-.5)*.35,vals,width=.35,label=g)
    axs[1,1].bar_label(bars,fontsize=9)
axs[1,1].set(xticks=np.arange(len(positions)),xticklabels=positions,ylabel='Sequences',title='Position and group (labels 1 and 2 only)')
axs[1,1].legend()
fig.suptitle('IRDS local data audit | observations, not model performance',fontsize=17)
fig.savefig(OUT/'data_overview.png',dpi=160); plt.close(fig)

# Choose a within-patient, same-position correct/incorrect pair: reduces identity and posture differences.
candidates=[]
for g in [2,3]:
    for s in subs:
        for pos in positions:
            a=[r for r in valid if r['subject']==s and r['gesture']==g and r['position']==pos and r['label']==1 and r['group']=='patient']
            b=[r for r in valid if r['subject']==s and r['gesture']==g and r['position']==pos and r['label']==2 and r['group']=='patient']
            if a and b:
                candidates.append((g,s,pos,a,b))
selected=[]
for g in [2,3]:
    options=[c for c in candidates if c[0]==g]
    if not options: continue
    c=max(options,key=lambda c:min(len(c[3]),len(c[4])))
    for rs in c[3:]:
        # Prefer fewer inferred joints, then a duration near the median of this subgroup.
        mid=np.median([r['frames'] for r in rs])
        selected.append(min(rs,key=lambda r:(r['tracking_states'].get('NotTracked',0)+r['tracking_states'].get('Inferred',0),abs(r['frames']-mid))))

fig,axs=plt.subplots(len(selected),3,figsize=(14,3.5*len(selected)),constrained_layout=True)
selected_details=[]
for rowidx,r in enumerate(selected):
    x=arrays[r['file']]
    # Root center each frame. This visualizes relative pose and removes global translation.
    root=x-x[:,0:1,:]
    elev_l=angle(x[:,5]-x[:,4],x[:,0]-x[:,20])
    elev_r=angle(x[:,9]-x[:,8],x[:,0]-x[:,20])
    active_right=elev_r.max()>elev_l.max()
    elev=elev_r if active_right else elev_l
    sh,el,wr=(8,9,10) if active_right else (4,5,6)
    elbow=angle(x[:,sh]-x[:,el],x[:,wr]-x[:,el])
    frames=np.unique(np.array([0,int(np.argmax(elev)),len(x)-1]))
    for ax,(dim,title) in zip(axs[rowidx,:2],[(0,'Front projection: X/Y'),(2,'Side projection: Z/Y')]):
        for k,fi in enumerate(frames):
            color=['#8aa0b8','#c66a40','#287f79'][k]
            for a,b in EDGES:
                ax.plot(root[fi,[a,b],dim],root[fi,[a,b],1],color=color,alpha=.75,lw=1.5)
            ax.scatter(root[fi,:,dim],root[fi,:,1],color=color,s=7,label=f'Frame {fi}')
        ax.set(title=title,aspect='equal',xlabel='Root-relative coordinate',ylabel='Root-relative Y')
        ax.legend(fontsize=7,loc='best')
    ax=axs[rowidx,2]
    t=np.arange(len(x))/30
    ax.plot(t,elev_l,label='Raw Left: arm / downward torso')
    ax.plot(t,elev_r,label='Raw Right: arm / downward torso')
    ax.plot(t,elbow,ls='--',label='Dominant arm: elbow interior angle')
    ax.set(xlabel='Nominal time (seconds)',ylabel='Geometric angle (degrees)',ylim=(0,185),
           title=f'ID {r["subject"]}, gesture {r["gesture"]}, label {r["label"]}, {r["position"]}')
    ax.legend(fontsize=7)
    selected_details.append({**r,'dominant_arm_raw_name':'Right' if active_right else 'Left',
                             'peak_upper_arm_torso_angle_degrees':float(elev.max()),
                             'elbow_angle_at_peak_elevation_degrees':float(elbow[int(np.argmax(elev))])})
fig.suptitle('Actual labeled examples | same patient and position within each gesture\nAngles are geometric proxies; no error subtype or clinical score is annotated.',fontsize=15)
fig.savefig(OUT/'shoulder_examples.png',dpi=150);plt.close(fig)
(OUT/'visualized_samples.json').write_text(json.dumps(selected_details,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['subject_position_label','joint_tracking_states','subject_label']},indent=2))
print('SELECTED',json.dumps(selected_details,indent=2))
