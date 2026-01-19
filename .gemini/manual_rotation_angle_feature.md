# 手动指定旋转角度功能

## 需求说明

### 背景
在现有的 03_regui 应用中，用户只能从不同 rotation angle 旋转角度计算出来的 RMSE 曲线中选择局部最小值（local minima）对应的角度作为 potential angle。对于一些特殊情况，这种方式不够灵活。

### 功能需求

1. **GUI 菜单**：在 Menu → Edit 中添加 "Rotation Angle" 子菜单，包含：
   - **Manual Specify...**: 弹出对话框，让用户输入 0-360 度之间（间隔 10 度）的 rotation angle 值，添加到 potential angle 列表中
   - **Reset Manual Angles**: 清除所有手动添加的 potential angle，恢复成默认生成的 local minimum angle 列表

2. **JSON 输出更新**：
   - 无手动指定时：
     ```json
     "rms_analysis": {
         "potential_local_minima": [260, 50, 120, 30, 100, 10, 330],
         "selected": 0
     }
     ```
   - 有手动指定时：
     ```json
     "rms_analysis": {
         "potential_local_minima": [260, 50, 120, 320, 30, 100, 10, 330],
         "manual_potential": [3],
         "selected": 3
     }
     ```
   - `manual_potential` 记录手动添加的角度在 `potential_local_minima` 列表中的索引

---

## 修改文件清单

### 1. core/alignment.py

**修改内容**：在 `AlignmentResult` 数据类中添加 `manual_potential_indices` 属性

```python
@dataclass
class AlignmentResult:
    # ... 其他属性 ...
    nuv_matrices: list = field(default_factory=list)
    manual_potential_indices: list = field(default_factory=list)  # 新增
```

**作用**：存储手动指定角度在 peak 列表中的索引位置

---

### 2. core/io_utils.py

**修改内容**：更新 `save_result_json()` 函数

```python
def save_result_json(
    # ... 其他参数 ...
    manual_potential_indices: Optional[list] = None,  # 新增参数
) -> None:
```

**实现细节**：
- 构建 `rms_analysis` 字典时，如果 `manual_potential_indices` 不为空，添加 `"manual_potential"` 字段
- 向后兼容：只有存在手动角度时才输出该字段

---

### 3. widgets/rmse_chart.py

**修改内容**：支持手动角度的可视化显示

1. **新增属性**：
   ```python
   self._manual_peak_flags: np.ndarray = np.array([], dtype=bool)
   ```

2. **更新 `set_data()` 方法**：
   ```python
   def set_data(
       self,
       angles: np.ndarray,
       rmses: np.ndarray,
       peaks: np.ndarray,
       selected: int = 0,
       manual_peak_flags: Optional[np.ndarray] = None,  # 新增参数
   ) -> None:
   ```

3. **更新 `_update_chart()` 方法**：
   - 手动角度：**橙色虚线** (orange, dashed, linewidth=1.5)
   - 自动检测角度：灰色实线 (gray, solid)
   - 当前选中角度：红色（如果是手动角度则为虚线）

---

### 4. ui/main_window.py

**修改内容**：添加菜单和处理逻辑

1. **新增状态变量**：
   ```python
   self._manual_specified_angles: list[int] = []
   ```

2. **更新 `_setup_menu()` 方法**：
   ```python
   # Rotation Angle 子菜单
   rotation_menu = QMenu("Rotation &Angle", self)
   edit_menu.addMenu(rotation_menu)

   manual_angle_action = QAction("&Manual Specify...", self)
   manual_angle_action.triggered.connect(self._on_manual_angle_specify)
   rotation_menu.addAction(manual_angle_action)

   reset_angles_action = QAction("&Reset Manual Angles", self)
   reset_angles_action.triggered.connect(self._on_reset_manual_angles)
   rotation_menu.addAction(reset_angles_action)
   ```

3. **新增处理方法**：

   - `_on_manual_angle_specify()`:
     - 显示下拉列表对话框（0, 10, 20, ..., 350）
     - 验证角度是否已存在（自动检测或手动）
     - 添加到 `_manual_specified_angles` 列表
     - 重新运行对齐并更新图表

   - `_on_reset_manual_angles()`:
     - 清空 `_manual_specified_angles` 列表
     - 重新运行对齐，恢复到只有自动检测的 peaks

4. **更新 `_on_item_selected()` 方法**：
   - 从已保存的 JSON 中读取 `manual_potential` 字段
   - 恢复 `_manual_specified_angles` 列表

5. **更新 `_run_alignment()` 方法**：
   - 将手动角度合并到 peak 列表
   - 构建 `manual_peak_flags` 数组
   - 传递给 RMSE 图表

6. **更新 `_save_result()` 方法**：
   - 传递 `manual_potential_indices` 参数给 `save_result_json()`

---

## 使用方法

1. **添加手动角度**：
   - 菜单：Edit → Rotation Angle → Manual Specify...
   - 从下拉列表选择角度（0°-350°，步长 10°）
   - 图表中会出现橙色虚线表示手动添加的角度

2. **选择手动角度**：
   - 使用 "Prev Peak" / "Next Peak" 按钮导航到手动角度
   - 或使用快捷键

3. **保存**：
   - 保存时会自动记录 `manual_potential` 字段
   - 下次加载时会自动恢复手动角度

4. **重置**：
   - 菜单：Edit → Rotation Angle → Reset Manual Angles
   - 清除所有手动添加的角度

---

## 视觉效果

| 类型 | 颜色 | 线型 |
|------|------|------|
| 自动检测 (未选中) | 灰色 | 实线 |
| 手动添加 (未选中) | 橙色 | 虚线 |
| 当前选中 (自动) | 红色 | 实线 |
| 当前选中 (手动) | 红色 | 虚线 |
