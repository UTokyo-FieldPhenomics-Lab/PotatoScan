# Peak Selection Logic Fix (2026-01-20)

## 问题描述

### 问题 1: Next Peak 按钮导航逻辑不一致
- **现象**: 点击 "Next Peak" 按钮时，3D 物体旋转看起来是从左到右的逻辑（按角度递增）
- **但是**: RMSE Analysis 可视化中的 "Current Peak" 标记是按照 RMSE 从小到大排序的
- **结果**: 用户困惑，导航顺序与视觉不一致

### 问题 2: 状态栏信息不够详细
- **需求**: 在底部状态栏显示更详细的信息，格式为 `0205-060: RMSE=xxx, (Peak 1: rotation angle)`

### 问题 3: 旧 JSON 文件兼容性
- **背景**: 旧的 output JSON 文件中 `potential_local_minima` 是按 RMSE 排序的角度列表
- **问题**: 移除 RMSE 排序后，`selected` 索引指向的角度会变化

---

## 解决方案

### 1. 移除 RMSE 排序，改为角度顺序

**文件**: `core/alignment.py`

**修改**: 在 `compute_nuv_rotation` 方法中移除了按 RMSE 排序的代码块：

```python
# 移除的代码:
# Sort peaks by RMSE values
if len(peaks) > 1:
    peak_values = rmses[peaks]
    order = np.argsort(peak_values)
    peaks = peaks[order]
```

**效果**: 现在 peaks 按它们在 RMSE 曲线中出现的位置排列（即按角度递增顺序），"Next Peak" 按钮会从左到右连续导航。

### 2. 自动选择最佳 Peak（最小 RMSE）

**文件**: `core/alignment.py`

**修改**: 更新 `compute_full_alignment` 方法，当 `selected_peak=None` 时自动选择 RMSE 最小的 peak：

```python
def compute_full_alignment(
    self,
    rgbd_data: dict,
    sfm_data: dict,
    selected_peak: Optional[int] = None,  # 改为 Optional[int]
) -> AlignmentResult:
    ...
    # Select peak
    peaks = nuv_result["peaks"]
    
    if selected_peak is None:
        # Default to peak with lowest RMSE
        if len(peaks) > 0:
            peak_values = nuv_result["rmses"][peaks]
            peak_idx = int(np.argmin(peak_values))
        else:
            peak_idx = 0
    else:
         peak_idx = min(selected_peak, len(peaks) - 1)
```

### 3. 使用角度值而非索引传递 Peak 选择

**文件**: `ui/main_window.py`

**核心改动**:
- 将 `_run_alignment` 的参数从 `selected_peak_idx: Optional[int]` 改为 `selected_peak_angle: Optional[float]`
- 加载旧文件时，从 `potential_local_minima[selected]` 获取实际角度值
- 在对齐计算后，通过 `np.isclose()` 在新的 peaks 列表中查找匹配角度

**加载逻辑**:
```python
# 从保存的数据中获取选中的 peak 角度
rms_analysis = meta.get("rms_analysis", {})
saved_peak_angles = rms_analysis.get("potential_local_minima", [])
saved_selected_idx = rms_analysis.get("selected", 0)

# 获取实际被选中的角度值
if saved_peak_angles and saved_selected_idx < len(saved_peak_angles):
    selected_peak_angle = saved_peak_angles[saved_selected_idx]
else:
    selected_peak_angle = None
```

**查找匹配角度**:
```python
# 如果指定了特定角度，查找其索引并重新计算
if selected_peak_angle is not None:
    peak_angles = self._current_result.peak_angles
    matching_indices = np.where(np.isclose(peak_angles, selected_peak_angle))[0]
    if len(matching_indices) > 0:
        target_peak_idx = int(matching_indices[0])
        self._current_result = self._aligner.recompute_with_peak(
            target_peak_idx,
            self._current_rgbd,
            self._current_sfm,
        )
```

### 4. 更新状态栏信息格式

**文件**: `ui/main_window.py`

**修改**: 在 `_run_alignment` 和 `_on_peak_changed` 中更新状态栏消息格式：

```python
# 获取选中 peak 的角度
current_idx = self._current_result.selected_peak_idx
if 0 <= current_idx < len(self._current_result.peak_angles):
    angle = self._current_result.peak_angles[current_idx]
    peak_info = f", (Peak {current_idx}: {angle}°)"
else:
    peak_info = ""

self._status_bar.showMessage(
    f"{self._current_pid}: RMSE={self._current_result.rmse:.6f}{peak_info}"
)
```

**输出示例**: `2025-060: RMSE=0.027911, (Peak 0: 50°)`

---

## 修改的文件列表

| 文件 | 修改类型 | 描述 |
|------|---------|------|
| `core/alignment.py` | 逻辑修改 | 移除 RMSE 排序，添加自动选择最佳 peak |
| `ui/main_window.py` | 接口修改 | `_run_alignment` 使用角度而非索引 |
| `ui/main_window.py` | 新增 import | 添加 `import numpy as np` 到顶部 |
| `ui/main_window.py` | 状态栏 | 更新消息格式包含 peak 索引和角度 |

---

## 兼容性保证

### 旧 JSON 文件格式
```json
"rms_analysis": {
    "potential_local_minima": [50, 220, 280, 340, 140],  // RMSE 排序的角度
    "selected": 0  // 索引 0 = 角度 50°
}
```

### 新逻辑处理流程
1. 读取 `potential_local_minima[selected]` 获取实际角度值（如 50°）
2. 执行对齐，计算新的 peaks（按角度顺序）
3. 在新 peaks 中查找 50° 对应的索引
4. 使用该索引重新计算对齐结果

### 结果
- ✅ 旧文件中保存的角度能正确恢复
- ✅ "Next/Prev Peak" 按角度顺序导航
- ✅ 初始加载自动选择最佳 RMSE 的 peak
- ✅ 状态栏显示详细的调试信息

---

## 修复的 Bug

1. **UnboundLocalError: 'np' not defined**
   - 原因: 函数内部有局部 `import numpy as np`，遮蔽了新添加的代码中使用的 `np`
   - 解决: 移除局部导入，在文件顶部统一导入

2. **UnboundLocalError: 'selected_peak_angle' not defined**
   - 原因: 当没有 output JSON 文件时，`selected_peak_angle` 从未被定义
   - 解决: 在 `if output_path.exists()` 之前初始化 `selected_peak_angle = None`
