#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLK Helper - V16  (Airtest 1.3.6 兼容版, RGB颜色匹配)
...（完整注释见文件头，省略此处解释以保持文件简洁）
"""
import os, sys, glob, json, time, logging
from typing import List, Tuple, Dict

import numpy as np
import cv2, pyautogui
from PyQt5 import QtCore, QtGui, QtWidgets

# 降噪 Airtest 内部日志：仅 ERROR
for name in ["airtest", "airtest.core", "airtest.aircv"]:
    logging.getLogger(name).setLevel(logging.ERROR)


GRID = 8
W_FIXED, H_FIXED = 932, 932
POLL_MS = 1000
DELAY_START_MS = 4000
TEMPLATE_SIZE = 96
EPS_STICK = 0.05
THRESH_CONF = 0.50
ALPHA_CELL = 0.30
BROWN_ALPHA = 0.60
TIP_THICK = 5
GRID_THICK = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, "templates")
CONFIG_PATH = os.path.join(SCRIPT_DIR, "match_overlay_config.json")

CN2EN = {"棕色":"Brown","紫色":"Purple","绿色":"Green","黄色":"Yellow","红色":"Red","蓝色":"Blue"}
COLOR_ORDER_CN_DEFAULT = ["棕色","紫色","绿色","黄色","红色","蓝色"]
ALL_COLOR_EN = ["Red","Purple","Green","Blue","Brown","Yellow","Skull"]
BALL_COLOR_EN = ["Red","Purple","Green","Blue","Brown","Yellow"]

QCOLOR_FOR = {
    "Red":    QtGui.QColor(255, 64, 64),
    "Purple": QtGui.QColor(170, 80, 255),
    "Green":  QtGui.QColor(80, 200, 120),
    "Blue":   QtGui.QColor(80, 150, 255),
    "Brown":  QtGui.QColor(120, 70, 30),
    "Yellow": QtGui.QColor(255, 220, 80),
    "Skull":  QtGui.QColor(220, 220, 220),
}

def dprint(enabled: bool, *args):
    if enabled:
        print(*args)

def imread_unicode(path: str, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None

def load_config():
    x, y = 494, 100
    pr = COLOR_ORDER_CN_DEFAULT[:]
    show = True
    debug = True
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            x = int(d.get("roi_x", x)); y = int(d.get("roi_y", y))
            pr0 = d.get("priority_cn", pr)
            seen=set(); pr=[]
            for c in pr0:
                if c in CN2EN and c not in seen: pr.append(c); seen.add(c)
            for c in COLOR_ORDER_CN_DEFAULT:
                if c not in seen: pr.append(c); seen.add(c)
            pr = pr[:6]
            show = bool(d.get("show_color_overlay", True))
            debug = bool(d.get("debug_enabled", True))
        except Exception:
            pass
    return (x, y), pr, show, debug

def save_config(x, y, pr, show, debug):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "roi_x": x, "roi_y": y,
            "priority_cn": pr[:6],
            "show_color_overlay": show,
            "debug_enabled": debug
        }, f, ensure_ascii=False, indent=2)

def split_grid(frame: np.ndarray):
    H, W = frame.shape[:2]
    cellH, cellW = H // GRID, W // GRID
    cells = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            y0, y1 = r * cellH, (r + 1) * cellH if r < GRID - 1 else H
            x0, x1 = c * cellW, (c + 1) * cellW if c < GRID - 1 else W
            cell = frame[y0:y1, x0:x1]
            cell = cv2.resize(cell, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
            row.append(cell)
        cells.append(row)
    return cells

NAME_ALIASES = {
    "red":"Red","blue":"Blue","green":"Green","yellow":"Yellow",
    "purple":"Purple","brown":"Brown","skull":"Skull"
}

class AirtestLightMatcher:
    def __init__(self, template_dir: str, debug: bool):
        self.debug = debug
        self.templates: Dict[str, np.ndarray] = {}
        self._load_templates(template_dir)
        self.prev_color = [[None]*GRID for _ in range(GRID)]
        self.prev_score = [[0.0]*GRID for _ in range(GRID)]

    def _load_templates(self, template_dir: str):
        paths = glob.glob(os.path.join(template_dir, "*.[pP][nN][gG]"))
        for p in paths:
            base = os.path.splitext(os.path.basename(p))[0].lower()
            name = NAME_ALIASES.get(base)
            if not name:
                continue
            img = imread_unicode(p, cv2.IMREAD_COLOR)
            if img is None:
                print("⚠️ 模板读取失败:", p)
                continue
            if img.shape[:2] != (TEMPLATE_SIZE, TEMPLATE_SIZE):
                img = cv2.resize(img, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
            self.templates[name] = img

        if not self.templates:
            print("❌ 未加载到任何模板。期望目录：", template_dir)
        else:
            print("✅ 模板加载完成：", sorted(self.templates.keys()))

    
        
    def _match_score(self, cell_bgr, tmpl_bgr):
        """
        颜色敏感的模板匹配函数（纯 OpenCV 版，无 Airtest 依赖）
        ------------------------------------------------------------
        参数:
            cell_bgr : np.ndarray - 当前棋盘单元格图像（BGR格式）
            tmpl_bgr : np.ndarray - 模板图像（BGR格式）
        返回:
            float - 匹配置信度 (0.0~1.0)
        """
        try:
            # 将两张图都从 BGR 转为 RGB，以启用颜色匹配
            cell_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
            tmpl_rgb = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2RGB)

            # 使用 OpenCV 三通道模板匹配（颜色敏感）
            res = cv2.matchTemplate(cell_rgb, tmpl_rgb, cv2.TM_CCOEFF_NORMED)

            # 提取最大匹配分数作为置信度
            conf = float(np.max(res))
            return conf
        except Exception as e:
            if self.debug:
                print("[match error]", e)
            return 0.0


    def classify_cell(self, cell_bgr: np.ndarray, r: int, c: int):
        best_name, best_score = None, -1.0
        if not self.templates:
            return "Skull", 0.0
        for name in BALL_COLOR_EN + ["Skull"]:
            tmpl = self.templates.get(name)
            if tmpl is None: 
                continue
            s = self._match_score(cell_bgr, tmpl)
            if s > best_score:
                best_score, best_name = s, name
        pn, ps = self.prev_color[r][c], self.prev_score[r][c]
        if pn is not None and (abs(ps - best_score) < EPS_STICK or best_score < THRESH_CONF):
            best_name, best_score = pn, ps
        self.prev_color[r][c], self.prev_score[r][c] = best_name, best_score
        return best_name or "Skull", float(best_score if best_score>0 else 0.0)

def classify_board_by_templates(frame: np.ndarray, matcher: AirtestLightMatcher):
    cells = split_grid(frame)
    labels = []
    for r in range(GRID):
        row = []
        for c in range(GRID):
            name, _ = matcher.classify_cell(cells[r][c], r, c)
            row.append(name)
        labels.append(row)
    return labels

def in_bounds(r, c): return 0 <= r < GRID and 0 <= c < GRID

def swap_simple(board, r1, c1, r2, c2):
    nb = [row[:] for row in board]
    nb[r1][c1], nb[r2][c2] = nb[r2][c2], nb[r1][c1]
    return nb

def find_runs_for_color(board, color):
    rh, rv = [], []
    for r in range(GRID):
        c = 0
        while c < GRID:
            if board[r][c] != color: 
                c += 1; continue
            c0 = c
            while c + 1 < GRID and board[r][c + 1] == color:
                c += 1
            if c - c0 + 1 >= 3:
                rh.append((r, c0, c))
            c += 1
    for c in range(GRID):
        r = 0
        while r < GRID:
            if board[r][c] != color: 
                r += 1; continue
            r0 = r
            while r + 1 < GRID and board[r + 1][c] == color:
                r += 1
            if r - r0 + 1 >= 3:
                rv.append((c, r0, r))
            r += 1
    return rh, rv

def union_mask_from_runs(rh, rv):
    m = [[False]*GRID for _ in range(GRID)]
    for (r, c0, c1) in rh:
        for c in range(c0, c1+1): 
            m[r][c] = True
    for (c, r0, r1) in rv:
        for r in range(r0, r1+1): 
            m[r][c] = True
    return m

def components_from_mask(mask):
    seen = [[False]*GRID for _ in range(GRID)]
    comps = []
    for r in range(GRID):
        for c in range(GRID):
            if mask[r][c] and not seen[r][c]:
                stack=[(r,c)]; seen[r][c]=True; comp=[(r,c)]
                while stack:
                    rr,cc=stack.pop()
                    for dr,dc in ((1,0),(0,1),(-1,0),(0,-1)):
                        nr,nc=rr+dr,cc+dc
                        if in_bounds(nr,nc) and mask[nr][nc] and not seen[nr][nc]:
                            seen[nr][nc]=True; stack.append((nr,nc)); comp.append((nr,nc))
                comps.append(comp)
    return comps

def analyze_swap(board, pr_map):
    best_sz=0; allc=[]
    for col in ALL_COLOR_EN:
        rh,rv=find_runs_for_color(board,col)
        if not rh and not rv: 
            continue
        m=union_mask_from_runs(rh,rv); cs=components_from_mask(m)
        for comp in cs:
            if len(comp)>=3:
                allc.append((len(comp),col,comp)); best_sz=max(best_sz,len(comp))
    if best_sz==0: return 0,0,[],[]
    if best_sz>=4:
        sel=[(sz,col,comp) for (sz,col,comp) in allc if sz==best_sz]
        paths=[comp for (_,_,comp) in sel]; cols=[col for (_,col,_) in sel]
        cp=max((pr_map.get(col,0) for col in cols),default=0)
        return best_sz,cp,paths,cols
    colors_3={col for (sz,col,_) in allc if sz==3}
    if not colors_3: return 0,0,[],[]
    best_color=max(colors_3,key=lambda c:pr_map.get(c,0))
    for (sz,col,comp) in allc:
        if sz==3 and col==best_color:
            return 3,pr_map.get(col,0),[comp],[col]
    return 0,0,[],[]

def choose_best_move(board, pr_cn, debug=False):
    pr_map={CN2EN[c]:(6-i) for i,c in enumerate(pr_cn)}
    best=(0,0,[],[],None); cnt=0
    for r in range(GRID):
        for c in range(GRID):
            for dr,dc in ((0,1),(1,0)):
                nr,nc=r+dr,c+dc
                if not in_bounds(nr,nc): continue
                if board[r][c]==board[nr][nc]: continue
                nb=swap_simple(board,r,c,nr,nc); cnt+=1
                sz,cp,ps,cs=analyze_swap(nb,pr_map)
                if sz>best[0] or (sz==best[0] and cp>best[1]): best=(sz,cp,ps,cs,((r,c),(nr,nc)))
    dprint(debug,f"[Choose] 评估交换={cnt}, best_size={best[0]}, score={best[1]}, colors={best[3]}")
    return best

class Overlay(QtWidgets.QWidget):
    def __init__(self, get_state):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._get_state=get_state
        self.resize(W_FIXED,H_FIXED)

    def set_topmost(self, topmost: bool):
        self.hide(); self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, topmost); self.show()

    def sync_pos(self):
        x,y=self._get_state()["roi_xy"]; self.move(x,y)

    def paintEvent(self,evt):
        st=self._get_state()
        labels = st.get("board_labels")
        paths  = st.get("tip_paths",[])
        show   = bool(st.get("show_color_overlay",True))

        p=QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing,True)
        w,h=self.width(),self.height(); cw,ch=w/GRID,h/GRID

        if show and labels is not None:
            for r in range(GRID):
                for c in range(GRID):
                    name=labels[r][c]
                    col=QtGui.QColor(QCOLOR_FOR.get(name,QtGui.QColor(0,0,0)))
                    col.setAlpha(int((BROWN_ALPHA if name=="Brown" else ALPHA_CELL)*255))
                    p.fillRect(QtCore.QRectF(c*cw,r*ch,cw,ch),col)

        # 顶部标签
        if labels is not None:
            font=QtGui.QFont("Microsoft YaHei", max(8,int(ch*0.22)))
            p.setFont(font)
            for r in range(GRID):
                for c in range(GRID):
                    name=labels[r][c]; text=name.lower()
                    rect=QtCore.QRectF(c*cw+2, r*ch+2, cw-4, ch*0.35)
                    p.setPen(QtGui.QPen(QtGui.QColor(0,0,0,220), 2)); p.drawText(rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, text)
                    p.setPen(QtGui.QPen(QtGui.QColor(255,255,255,255), 1)); p.drawText(rect, QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter, text)

        pen_grid=QtGui.QPen(QtGui.QColor(255,0,0,255),GRID_THICK); pen_grid.setCapStyle(QtCore.Qt.FlatCap)
        p.setPen(pen_grid)
        for i in range(GRID+1):
            x=int(round(i*cw)); y=int(round(i*ch))
            p.drawLine(x,0,x,h); p.drawLine(0,y,w,y)

        if paths:
            pen_tip=QtGui.QPen(QtGui.QColor(255,255,255,255),TIP_THICK)
            pen_tip.setCapStyle(QtCore.Qt.RoundCap); pen_tip.setJoinStyle(QtCore.Qt.RoundJoin)
            p.setPen(pen_tip)
            for comp in paths:
                s=set(comp)
                for (r,c) in comp:
                    x1,y1=(c+0.5)*cw,(r+0.5)*ch
                    for dr,dc in ((1,0),(0,1)):
                        if (r+dr,c+dc) in s:
                            x2,y2=(c+dc+0.5)*cw,(r+dr+0.5)*ch
                            p.drawLine(QtCore.QPointF(x1,y1),QtCore.QPointF(x2,y2))

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("连连看提示器 - 控制面板 (LLK Helper V16)")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowMinimizeButtonHint)
        self.resize(660, 600)

        (x0,y0), pr0, show0, debug0 = load_config()

        tip=QtWidgets.QLabel("识别每1秒；点击运行后延迟4秒；RGB颜色匹配；关键帧日志可开关。")
        tip.setWordWrap(True)

        self.edit_x, self.edit_y = QtWidgets.QLineEdit(str(x0)), QtWidgets.QLineEdit(str(y0))
        self.edit_x.setValidator(QtGui.QIntValidator(0,10000))
        self.edit_y.setValidator(QtGui.QIntValidator(0,10000))
        xy=QtWidgets.QGridLayout()
        xy.addWidget(QtWidgets.QLabel("起始 X："),0,0); xy.addWidget(self.edit_x,0,1)
        xy.addWidget(QtWidgets.QLabel("起始 Y："),1,0); xy.addWidget(self.edit_y,1,1)

        colors_cn=list(CN2EN.keys())
        self.combos=[]; grid=QtWidgets.QGridLayout()
        for i in range(6):
            lbl=QtWidgets.QLabel(f"优先级{i+1}："); cb=QtWidgets.QComboBox(); cb.addItems(colors_cn)
            self.combos.append(cb); grid.addWidget(lbl,i,0); grid.addWidget(cb,i,1)
        for i,c in enumerate(pr0):
            if i<6 and c in colors_cn: self.combos[i].setCurrentText(c)

        self.chk_fill  = QtWidgets.QCheckBox("显示颜色识别填充（调试）"); self.chk_fill.setChecked(show0)
        self.chk_debug = QtWidgets.QCheckBox("调试输出日志（Console）"); self.chk_debug.setChecked(True)

        self.btn_run, self.btn_stop = QtWidgets.QPushButton("▶ 运行（4秒后开始）"), QtWidgets.QPushButton("⏹ 停止")
        self.btn_stop.setEnabled(False)
        hb=QtWidgets.QHBoxLayout(); hb.addWidget(self.btn_run); hb.addWidget(self.btn_stop)

        lay=QtWidgets.QVBoxLayout(self)
        lay.addWidget(tip); lay.addLayout(xy); lay.addLayout(grid); lay.addWidget(self.chk_fill); lay.addWidget(self.chk_debug); lay.addLayout(hb)

        self.roi_xy=(x0,y0)
        self.priority_cn=self._get_priority()
        self.show_color_overlay=self.chk_fill.isChecked()
        self.debug_enabled=True
        self.board_labels=None; self.tip_paths=[]; self.running=False

        self.matcher=AirtestLightMatcher(TEMPLATE_DIR, debug=self.debug_enabled)
        self.overlay=Overlay(self._get_state); self.overlay.hide()
        self.timer=QtCore.QTimer(self); self.timer.setInterval(POLL_MS); self.timer.timeout.connect(self._tick)

        self.btn_run.clicked.connect(self.on_run); self.btn_stop.clicked.connect(self.on_stop)
        self.chk_fill.stateChanged.connect(self.on_toggle_fill); self.chk_debug.stateChanged.connect(self.on_toggle_debug)

        if not self.matcher.templates:
            QtWidgets.QMessageBox.warning(self, "模板未就绪", f"未在目录加载到模板：\n{TEMPLATE_DIR}\n需要 red/blue/green/yellow/purple/brown/skull.png")

    def _get_state(self): 
        return {"roi_xy":self.roi_xy,"board_labels":self.board_labels,"tip_paths":self.tip_paths,"show_color_overlay":self.show_color_overlay}

    def _get_priority(self)->List[str]:
        pr=[cb.currentText() for cb in self.combos]; seen=set(); out=[]
        for c in pr:
            if c not in seen: out.append(c); seen.add(c)
        for c in COLOR_ORDER_CN_DEFAULT:
            if c not in seen: out.append(c); seen.add(c)
        return out[:6]

    def on_toggle_fill(self,_):
        self.show_color_overlay=self.chk_fill.isChecked()
        save_config(self.roi_xy[0], self.roi_xy[1], self._get_priority(), self.show_color_overlay, self.debug_enabled)
        self.overlay.update()

    def on_toggle_debug(self,_):
        self.debug_enabled=self.chk_debug.isChecked()
        self.matcher.debug=self.debug_enabled
        save_config(self.roi_xy[0], self.roi_xy[1], self._get_priority(), self.show_color_overlay, self.debug_enabled)
        dprint(self.debug_enabled, "[Debug] 调试输出已开启" if self.debug_enabled else "[Debug] 调试输出已关闭")

    def on_run(self):
        try:
            x=int(self.edit_x.text().strip()); y=int(self.edit_y.text().strip())
        except ValueError:
            QtWidgets.QMessageBox.warning(self,"输入错误","请输入整数坐标。"); return
        self.roi_xy=(x,y)
        self.priority_cn=self._get_priority()
        self.show_color_overlay=self.chk_fill.isChecked()
        self.debug_enabled=self.chk_debug.isChecked()
        save_config(x,y,self.priority_cn,self.show_color_overlay,self.debug_enabled)

        if not self.matcher.templates:
            QtWidgets.QMessageBox.warning(self,"模板未就绪",f"目录中未加载到模板：\n{TEMPLATE_DIR}")
            print("❌ 无模板，暂停运行。"); return
        else:
            print("✅ 模板就绪：", sorted(self.matcher.templates.keys()))

        QtWidgets.QMessageBox.information(self,"即将开始",f"请在 {DELAY_START_MS//1000} 秒内切换到游戏窗口。")
        self.showMinimized()
        QtCore.QTimer.singleShot(DELAY_START_MS, self._start_after_delay)

    def _start_after_delay(self):
        self.overlay.sync_pos(); self.overlay.show(); self.overlay.set_topmost(True)
        self.running=True; self.timer.start()
        self.btn_run.setEnabled(False); self.btn_stop.setEnabled(True)
        dprint(self.debug_enabled, "[Run] 已开始轮询")

    def on_stop(self):
        self.running=False; self.timer.stop(); self.overlay.hide()
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        self.showNormal(); self.raise_(); self.activateWindow()
        dprint(self.debug_enabled, "[Run] 已停止")

    def showEvent(self, e: QtGui.QShowEvent):
        super().showEvent(e)
        if self.overlay.isVisible():
            self.overlay.set_topmost(False)

    def _tick(self):
        if not self.running: return
        try:
            self.overlay.hide()
            img = pyautogui.screenshot(region=(self.roi_xy[0], self.roi_xy[1], W_FIXED, H_FIXED))
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            self.overlay.show()

            if frame is None or frame.size == 0:
                print("❌ 截图失败：frame为空。请检查 ROI 坐标/权限。"); return

            t0=time.time()
            self.board_labels=classify_board_by_templates(frame,self.matcher)
            sz,cp,ps,cs,sw=choose_best_move(self.board_labels,self.priority_cn,debug=False)
            self.tip_paths=ps or []
            self.overlay.update()

            elapsed=int(1000*(time.time()-t0))
            dprint(self.debug_enabled, f"[Tick] 帧完成 用时={elapsed}ms 提示线={len(self.tip_paths)} 规模={sz} 颜色={cs}")
        except Exception as e:
            print("Tick error:", e)

def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app=QtWidgets.QApplication(sys.argv)
    w=MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
