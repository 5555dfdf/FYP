"""Illustrative 33-landmark forward kinematics; not a measured patient or model output."""
import json
import math
from pathlib import Path

import numpy as np
from export_data import EXERCISES, LABELS

HERE=Path(__file__).resolve().parent


def pose(exercise,code,t):
    a=(1-math.cos(2*math.pi*t))/2
    p=np.zeros((33,3),dtype=float)
    p[23]=[-.18,-1.9,0];p[24]=[.18,-1.9,0]
    p[11]=[-.32,-2.9,0];p[12]=[.32,-2.9,0]
    p[0]=[0,-3.43,-.11]
    p[1]=[-.045,-3.48,-.09];p[2]=[-.07,-3.48,-.075];p[3]=[-.095,-3.47,-.045]
    p[4]=[.045,-3.48,-.09];p[5]=[.07,-3.48,-.075];p[6]=[.095,-3.47,-.045]
    p[7]=[-.13,-3.43,0];p[8]=[.13,-3.43,0];p[9]=[-.035,-3.36,-.1];p[10]=[.035,-3.36,-.1]
    for x,knee,ankle,heel,toe in [(-.20,25,27,29,31),(.20,26,28,30,32)]:
        p[knee]=[x,-.98,0];p[ankle]=[x,-.12,0];p[heel]=[x,-.04,.08];p[toe]=[x,0,-.22]
    if code=='ERR_ELEVATED_SHOULDER':p[12,1]-=.22*a
    p[13]=p[11]+[-.035,.60,0];p[15]=p[13]+[-.015,.56,0]
    if exercise==7:
        angle=math.radians(38)*a
        upper=np.array([.015,math.cos(angle),math.sin(angle)])
        if code=='ERR_ARM_TO_SIDE':upper[0]+=.65*a
        upper/=np.linalg.norm(upper)
        fore=np.array([upper[0],math.cos(angle-math.radians(82)*a if code=='ERR_BENT_ARM' else angle),
                       math.sin(angle-math.radians(82)*a if code=='ERR_BENT_ARM' else angle)])
        if code=='ERR_ARM_TO_SIDE':fore=upper.copy()
    else:
        angle=math.radians(65 if code=='ERR_LOW_ABDUCTION' else 130)*a
        upper=np.array([math.sin(angle),math.cos(angle),-.72*a if code=='ERR_ARM_FORWARD' else 0])
        upper/=np.linalg.norm(upper)
        fa=angle-math.radians(72)*a if code=='ERR_BENT_ARM' else angle
        fore=np.array([math.sin(fa),math.cos(fa),-.72*a if code=='ERR_ARM_FORWARD' else 0])
    fore/=np.linalg.norm(fore)
    p[14]=p[12]+upper*.60;p[16]=p[14]+fore*.56
    for wrist,elbow,pinky,index,thumb,sign in [(15,13,17,19,21,-1),(16,14,18,20,22,1)]:
        direction=p[wrist]-p[elbow];direction/=np.linalg.norm(direction)
        p[pinky]=p[wrist]+direction*.11+np.array([sign*.035,0,.01])
        p[index]=p[wrist]+direction*.13+np.array([-sign*.02,0,-.01])
        p[thumb]=p[wrist]+direction*.06+np.array([-sign*.065,0,-.035])
    if code=='ERR_TORSO_FORWARD':
        angle=math.radians(21)*a;c,s=math.cos(angle),math.sin(angle)
        v=p[:23]-np.array([0,-1.9,0])
        v[:,[1,2]]=v[:,[1,2]]@np.array([[c,s],[-s,c]])
        p[:23]=v+np.array([0,-1.9,0])
    if code=='ERR_TORSO_ROTATION':
        angle=math.radians(38)*a;c,s=math.cos(angle),math.sin(angle)
        p[:23,[0,2]]=p[:23,[0,2]]@np.array([[c,s],[-s,c]])
    if code=='ERR_HEAD_TO_SIDE':
        angle=math.radians(24)*a;c,s=math.cos(angle),math.sin(angle)
        v=p[:11]-np.array([0,-3.05,0])
        v[:,[0,1]]=v[:,[0,1]]@np.array([[c,s],[-s,c]])
        p[:11]=v+np.array([0,-3.05,0])
    return np.round(p,5).tolist()


def main():
    source={'exercises':{str(ex):{**settings,'clips':{
        code:{'label':code,'name':'标准动作' if code=='correct' else LABELS[code][0],
              'hint':settings['hint'] if code=='correct' else LABELS[code][1],
              'highlight':[] if code=='correct' else LABELS[code][2]}
        for code in ['correct']+settings['errors']}} for ex,settings in EXERCISES.items()}}
    out={'source':'schematic','person':None,'frames':96,'duration':5.0,'processing':'illustrative 33-landmark kinematic poses; no clinical angle prescription','exercises':{}}
    for ex_id,ex in source['exercises'].items():
        result={k:v for k,v in ex.items() if k!='clips'}
        result.update(schematic=True,clips={})
        for code,clip in ex['clips'].items():
            result['clips'][code]={**{k:v for k,v in clip.items() if k not in ['points','id']},'id':f'示意 / {ex["name"]} / {clip["name"]}',
                'points':[pose(int(ex_id),code,i/95) for i in range(96)]}
        out['exercises'][ex_id]=result
    (HERE/'schematic_data.json').write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print('12 illustrative 33-landmark clips built')


if __name__=='__main__':main()
