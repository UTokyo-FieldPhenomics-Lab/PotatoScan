重构软件的UI和流程：

MainWindow的 _on_item_selected中，点击某个item，完成下面的步骤来逐步实现整体的流程：

Tab 1 (step 1) raw. 读取和展示原始点云，

Tab 2 (step 2) sfm pin seg. 使用参数来控制sfm pin的分割。
需要涉及到的控制参数：pin_segment.py 里的get函数有一些参数（thresh, nb_points, radius），是用来控制pin 分割的，应该读取到raw点云切换到tab 2后，才进行处理进而得到pin的分割和可视化结果。这个tab的输出才是dataloader中load_sfm的结果dict

Tab 3 (step 3) pin detection visualize. 可视化sfm pin 和 rgbd pin的分割+圆拟合+法向量结果
里面涉及到使用 util_pc.find_pin_center函数得到上述结果并可视化

Tab 4 (step 4) Pin neighbour. 使用参数来控制rgbd pin的分割。
需要涉及到的控制参数：pin_segment.py 里的get函数有一些参数（thresh, nb_points, radius），是用来控制pin 分割的，应该读取到raw点云切换到tab 2后，才进行处理进而得到pin的分割和可视化结果。这个tab的输出才是dataloader中load_sfm的结果dict

Tab 5 (step 5) cross-strip rough align: 使用rmse的local minimum来控制旋转角度，得到初始Tmatrix； 

Tab 6 (step 6) colored-icp refine: 基于coloricp的迭代数，进一步微调 Tmatrix，得到最终结果。

因此UI和处理逻辑还需要进一步调整：

UI的左下Parameters组，需要拆分后，移入到具体的tab中，每个tab由3Dviewer + 底部paramters控制组成（目前的RMSE analysis的位置），切换tab就等于切换底部的控制参数面板


Tab 3底部