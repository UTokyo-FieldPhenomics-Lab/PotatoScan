# DBSCAN Denoising Enhancement for SfM Pin Segmentation

## 概述

本功能增强了SfM点云的pin分割去噪能力，在`remove_radius_outlier`去噪后如果凸包体积仍超标，自动启动DBSCAN聚类来剔除远距离离群点。

## 问题背景

原有的去噪方式 `remove_radius_outlier(nb_points=min(40, pin_pcd_num/20), radius=0.005)` 存在问题：
- 对于只有70多个点的点云，`nb_points = 70/20 = 3`
- 如果噪声点附近有3+个邻居，它就不会被移除
- 导致远距离离群点无法被有效剔除，凸包体积过大

## 解决方案

### 1. DBSCAN聚类去噪 (`utils/pin_segment.py`)

**新增 `dbscan_denoise()` 方法：**
```python
@staticmethod
def dbscan_denoise(pin_pcd, eps: float = 0.005, min_samples: int = 3):
    """
    Apply DBSCAN clustering to remove distant outlier points.
    Returns: (kept_pcd, kept_indices, outlier_indices, dbscan_activated)
    """
```

**集成到迭代去噪循环：**
- 在 `remove_radius_outlier` 后检查凸包体积
- 如果仍超过目标体积，尝试DBSCAN聚类 (eps=0.008, min_samples=3)
- 保留最大聚类，剔除小聚类和噪声点
- 跟踪 `dbscan_outlier_idx` 和 `dbscan_activated` 标志

**输出结果增加字段：**
```python
results_container = {
    # ... existing fields ...
    "dbscan_outlier_idx": dbscan_outlier_idx,  # 被剔除点的索引
    "dbscan_activated": dbscan_activated,       # DBSCAN是否被激活
}
```

### 2. Tab 2 可视化 (`widgets/viewer_3d.py`)

在 `_update_sfm_pin_view()` 中添加橙色离群点显示：
```python
# Show DBSCAN outliers in orange (if any)
if dbscan_outlier_idx is not None and len(dbscan_outlier_idx) > 0:
    outlier_pcd = sfm_pcd_full.select_by_index(dbscan_outlier_idx)
    outlier_pcd.paint_uniform_color([1.0, 0.65, 0.0])  # Orange #FFA500
    self._add_mesh_to_plotter(self._plotter_sfm_pin, outlier_pcd, point_size=5)
```

**可视化布局（中间点云）：**
- 红色: 保留的pin点
- 橙色: DBSCAN剔除的离群点

### 3. Current Threshold 标签 (`widgets/parameter_panel.py`)

更新 `set_current_threshold()` 方法：
```python
def set_current_threshold(self, value: float, dbscan_activated: bool = False):
    suffix = " (DBSCAN activated)" if dbscan_activated else ""
    self._lbl_current_thresh.setText(f"{value:.2f}{suffix}")
```

显示格式: `0.35 (DBSCAN activated)`

### 4. 切换Item时的行为 (`ui/main_window.py`)

在 `_on_item_selected()` 中：
```python
# Reset ALL Steps (2, 3, 4) for new item
self._param_panel.reset_all_steps()

# Clear viewer and chart for fresh state
self._viewer.clear()
self._rmse_chart.clear()
```

新增 `reset_all_steps()` 方法复位所有步骤参数。

### 5. Threshold回调更新

更新回调签名以传递DBSCAN状态：
```python
def _on_threshold_update(self, threshold: float, dbscan_activated: bool = False):
    self._param_panel.set_current_threshold(threshold, dbscan_activated)
```

## 修改文件列表

| 文件 | 修改内容 |
|------|---------|
| `utils/pin_segment.py` | 添加DBSCAN导入、`dbscan_denoise()`方法、集成到迭代循环、输出字段 |
| `widgets/viewer_3d.py` | Tab 2中橙色离群点可视化 |
| `widgets/parameter_panel.py` | `set_current_threshold()`加DBSCAN后缀、`reset_all_steps()`方法 |
| `ui/main_window.py` | 切换item时清除viewer/chart、复位所有参数、更新threshold回调 |

## 测试结果 (2025-055)

```
DBSCAN found 2 clusters + noise
Cluster sizes: {0: 65, 1: 7}, keeping cluster 0
DBSCAN: kept 65 points, removed 7 outliers
DBSCAN reduced hull to 52.1mm³
```

- 原始: 73点, 凸包1999.2mm³
- radius_outlier后: 凸包1370.1mm³ (仍超标)
- DBSCAN后: 65点, 凸包52.1mm³ ✅

## Auto Iteration 复选框功能

同时实现了Auto Iteration复选框功能：

1. **`SfMPinParams`** 添加 `auto_iteration: bool = True` 字段
2. **Parameter Panel** 添加 "Auto Iteration" 复选框（默认勾选）
3. **Preview Mode**: 当迭代失败时自动进入预览模式：
   - 复选框自动取消勾选
   - 只显示 Tab 2 的可视化
   - 暂停 Tab 3/4 和 RMSE 分析
   - 用户可调整 Initial Threshold 实时预览分割效果
