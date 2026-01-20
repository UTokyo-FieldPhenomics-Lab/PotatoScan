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

---

## 追加修复: 手动角度插入后排序问题 (15:48)

### 问题描述
手动添加 rotation angle 后，由于角度被 append 到列表末尾，导致 "Prev Peak" 和 "Next Peak" 按钮按列表顺序而非角度顺序导航。

**例如**:
- 自动检测的 peaks: `[50, 140, 220]`
- 手动添加角度 `100°` 后: `[50, 140, 220, 100]`（错误顺序）
- 点击 "Next Peak" 会从 220° 跳到 100°，而不是按 50→100→140→220 的顺序

### 解决方案

**文件**: `ui/main_window.py`

**修改 `_run_alignment` 中的手动角度合并逻辑**:

```python
# 1. 收集手动角度对应的索引
manual_angle_indices = []
for ma in self._manual_specified_angles:
    manual_idx = int(ma / 10) - 1
    if 0 <= manual_idx < len(angles) and manual_idx not in peak_indices:
        peak_indices.append(manual_idx)
        manual_angle_indices.append(manual_idx)

# 2. 按角度值排序
peak_indices.sort(key=lambda idx: angles[idx])

# 3. 重建手动 peak 的索引（在排序后的列表中的位置）
manual_potential_indices = []
for i, peak_idx in enumerate(peak_indices):
    if peak_idx in manual_angle_indices:
        manual_potential_indices.append(i)

# 4. 查找选中角度在排序后列表中的索引
if selected_peak_angle is not None:
    target_idx = int(selected_peak_angle / 10) - 1
    for i, peak_idx in enumerate(peak_indices):
        if peak_idx == target_idx:
            chart_selected_idx = i
            break
```

**修改 `_on_manual_angle_specify`**:
- 移除了旧的 `new_peak_idx` 计算逻辑（不再需要，因为 `_run_alignment` 会自动处理排序和选择）

### 结果
- ✅ 手动添加的角度会被插入到正确的排序位置
- ✅ "Next/Prev Peak" 始终按角度顺序导航
- ✅ 手动添加后自动跳转到该角度

---

## 追加修复: 手动角度对齐计算问题 (15:53)

### 问题描述
手动设定完 rotation angle 点击确定后，RMSE Analysis 页面的图更新了，但 Tab4 Aligned 的 3D viewer 没有发生变化（需要手动 prev peak + next peak 两次切换才能正确显示）。

**原因**: 手动角度不在 `peak_angles`（自动检测的 peaks）中，所以 `_run_alignment` 中的 `matching_indices` 为空，导致 fallback 到自动选择的最佳 peak，而不是使用手动指定的角度计算对齐。

### 解决方案

**添加 `_recompute_with_manual_angle` 方法**:

```python
def _recompute_with_manual_angle(self, angle: float):
    """
    Compute alignment for a manually specified angle.
    Uses the cached NUV matrices to compute alignment for an angle
    that may not be in the auto-detected peaks list.
    """
    last = self._current_result
    
    # Find the index for this angle in the NUV matrices
    angle_idx = int(angle / 10) - 1
    nuv_matrix = last.nuv_matrices[angle_idx]

    # Recompute rough alignment + ICP
    imatrix = self._aligner.compute_rough_alignment(...)
    iimatrix = nuv_matrix @ imatrix
    tmatrix, o3d_rmse = self._aligner.compute_icp_refinement(...)
    
    return AlignmentResult(transform_matrix=tmatrix, ...)
```

**修改 `_run_alignment` 逻辑**:
```python
if len(matching_indices) > 0:
    # 在自动检测的 peaks 中找到匹配
    self._current_result = self._aligner.recompute_with_peak(...)
elif selected_peak_angle in self._manual_specified_angles:
    # 手动角度 - 使用直接角度索引计算
    self._current_result = self._recompute_with_manual_angle(selected_peak_angle)
else:
    # 警告并使用自动选择
    logger.warning(...)
```

### 结果
- ✅ 手动角度正确计算对齐
- ✅ 3D viewer 立即更新显示正确的变换
- ✅ 无需手动 prev/next peak 切换

---

## 追加修复: 修改参数重置手动角度问题 (16:04)

### 问题描述
手动设置过 angle 后，修改 Step 4 iteration 等参数会导致 alignment 重置为自动选择的 peak，而不是保持当前的手动角度。

### 原因
`_on_params_changed` 方法只检查 `self._current_result.peak_angles`（自动检测的 peaks），如果不匹配（手动角度索引超出范围），则强制重置为 `None`（自动选择最佳 peak）。

### 解决方案

引入 `self._current_selected_angle` 变量 explicitly 追踪当前激活的角度（无论是自动还是手动）。

1. **`__init__`**: 初始化 `self._current_selected_angle = None`
2. **`_run_alignment`**:
   - 默认设置为自动选择的最佳 peak angle
   - 如果成功应用了 `selected_peak_angle`（无论是自动还是手动），则更新 `self._current_selected_angle` 为该值
3. **`_on_params_changed`**:
   - 使用 `self._current_selected_angle` 而不是从 `selected_peak_idx` 推导
4. **`_on_item_selected`**: 重置 `self._current_selected_angle = None`

### 结果
- ✅ 修改参数时保持当前选中的手动角度
- ✅ 切换 Item 时正确重置状态

---

## UI 改进: Chart Legend for Manual Peak (16:17)

### 需求
当选取了 manual 设置的 angle 时，Legend 中应保留 manual 的说明。

### 修改
已在 `widgets/rmse_chart.py` 更新逻辑：
如果当前选中的 peak 是 manual 类型，Legend 标签显示为 **"Current (Manual)"**，且保持虚线样式。

```python
label="Current (Manual)" if is_manual else "Current"
```

---

## 追加修复: Prev/Next Peak 导航后参数更新问题 (16:47)

### 问题描述
上一个修复引入了新 bug：当没有手动角度时，使用 Prev/Next Peak 按钮切换 peak 后，修改参数（如 iteration）会导致 peak 重置回初始自动选择的 peak。

### 原因分析
`_on_peak_changed` 方法（处理 Prev/Next Peak 按钮点击）没有更新 `_current_selected_angle`，导致：
1. 加载 item → `_current_selected_angle = 50°`（自动选择）
2. 点击 "Next Peak" → 切换到 140°，但 `_current_selected_angle` 仍然是 50°
3. 修改参数 → 使用 `_current_selected_angle`（50°）→ 重置回 50°

### 解决方案
重写 `_on_peak_changed` 方法：
1. 从 chart 获取合并后的 peak 列表（包含 auto + manual）
2. 根据 peak_idx 获取实际角度值
3. **更新 `_current_selected_angle`**
4. 根据是 manual 还是 auto peak，调用相应的重计算方法

### 结果
- ✅ Prev/Next Peak 按钮正确更新 `_current_selected_angle`
- ✅ 修改参数后保持当前导航到的 peak
- ✅ 支持 auto 和 manual peaks
