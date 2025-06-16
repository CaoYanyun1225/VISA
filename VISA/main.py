import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import os


class MergedVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comprehensive Data Visualization & Analysis")
        self.root.geometry("1800x1000")

        # 数据存储 - 合并两个应用的数据变量
        self.npz_data = None
        self.npy_data = None
        self.arr_0 = None  # (sample, shape_number, shape_length)
        self.arr_1 = None  # (sample, shape_number, VP)
        self.x_train = None  # (sample, length, dimension_number)

        # 高级可视化数据
        self.heatmap_data = None  # (sample, shape_number, shape_number)
        self.attention_data = None  # (sample_number, shape_number, value_number)
        self.sorted_attention_data = None  # 排序后的数据
        self.original_indices = None  # 原始索引
        self.all_samples_indices = []  # 所有sample的排序索引

        # 可视化相关变量
        self.current_zoom = 1.0
        self.pan_offset = [0, 0]

        # Heatmap相关变量
        self.heatmap_colorbar = None
        self.current_heatmap_ax = None

        # 当前选中的shape信息
        self.selected_shapes = {'shape1': None, 'shape2': None}
        self.current_click_count = 0

        # Attention plot相关
        self.attention_annotations = []  # 存储注释对象

        # 序列控制变量
        self.sequence_controls = []

        # 创建主要布局
        self.create_main_layout()

    def create_main_layout(self):
        """创建主要布局"""
        # 创建主要的分割布局
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧菜单栏
        self.create_left_menu(main_paned)

        # 右侧可视化区域
        self.create_right_visualization(main_paned)

    def create_left_menu(self, parent):
        """创建左侧菜单栏"""
        menu_frame = ttk.Frame(parent, width=500)
        parent.add(menu_frame, weight=1)

        # 使用Canvas和Scrollbar创建可滚动的菜单
        canvas = tk.Canvas(menu_frame)
        scrollbar = ttk.Scrollbar(menu_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 创建标签页
        notebook = ttk.Notebook(scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 第一个标签页：基础可视化
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Visualization")
        self.create_basic_controls(basic_frame)

        # 第二个标签页：高级可视化
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced Analysis")
        self.create_advanced_controls(advanced_frame)

    def create_basic_controls(self, parent):
        """创建基础可视化控制"""
        # 文件加载部分
        self.create_file_loading_section(parent)

        # 上半部分可视化控制
        self.create_upper_viz_controls(parent)

        # 控制按钮
        self.create_control_buttons(parent)

        # Shape位置查看功能
        self.create_shape_position_controls(parent)

    def create_advanced_controls(self, parent):
        """创建高级可视化控制"""
        # Heatmap控制
        self.create_heatmap_controls(parent)

        # Attention控制
        self.create_attention_controls(parent)

    def create_file_loading_section(self, parent):
        """创建文件加载部分"""
        section1 = ttk.LabelFrame(parent, text="Data File Loading", padding=10)
        section1.pack(fill=tk.X, padx=5, pady=5)

        # NPZ文件选择
        ttk.Label(section1, text="Time Series Shapes Loading:").pack(anchor=tk.W)
        npz_frame = ttk.Frame(section1)
        npz_frame.pack(fill=tk.X, pady=2)

        self.npz_path_var = tk.StringVar()
        self.npz_entry = ttk.Entry(npz_frame, textvariable=self.npz_path_var, state="readonly")
        self.npz_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(npz_frame, text="Browse", command=self.browse_npz_file).pack(side=tk.RIGHT, padx=(5, 0))

        # NPY文件选择
        ttk.Label(section1, text="Time Series Raw Data Loading:").pack(anchor=tk.W, pady=(10, 0))
        npy_frame = ttk.Frame(section1)
        npy_frame.pack(fill=tk.X, pady=2)

        self.npy_path_var = tk.StringVar()
        self.npy_entry = ttk.Entry(npy_frame, textvariable=self.npy_path_var, state="readonly")
        self.npy_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(npy_frame, text="Browse", command=self.browse_npy_file).pack(side=tk.RIGHT, padx=(5, 0))

        # 加载按钮
        ttk.Button(section1, text="Load Data Files", command=self.load_data).pack(pady=(10, 0))

        # 数据信息显示
        self.data_info_text = tk.Text(section1, height=4, width=40)
        self.data_info_text.pack(pady=(5, 0), fill=tk.X)
        self.data_info_text.insert(tk.END, "No dataset loaded")
        self.data_info_text.config(state=tk.DISABLED)

    def create_upper_viz_controls(self, parent):
        """创建上半部分可视化控制"""
        section2 = ttk.LabelFrame(parent, text="Time Series Visualization Controls", padding=10)
        section2.pack(fill=tk.X, padx=5, pady=5)

        # 图片数量选择
        ttk.Label(section2, text="Number of Plots to Display:").pack(anchor=tk.W)
        self.plot_count_var = tk.IntVar(value=2)
        plot_count_frame = ttk.Frame(section2)
        plot_count_frame.pack(fill=tk.X, pady=2)
        for i in [1, 2, 3, 4]:
            ttk.Radiobutton(plot_count_frame, text=str(i), variable=self.plot_count_var,
                            value=i, command=self.update_sequence_controls).pack(side=tk.LEFT, padx=5)

        # 创建可滚动的序列参数控制区域
        self.sequence_control_canvas = tk.Canvas(section2, height=200)  # 重命名变量
        seq_scrollbar = ttk.Scrollbar(section2, orient="vertical", command=self.sequence_control_canvas.yview)
        self.sequence_frame = ttk.Frame(self.sequence_control_canvas)

        self.sequence_frame.bind(
            "<Configure>",
            lambda e: self.sequence_control_canvas.configure(scrollregion=self.sequence_control_canvas.bbox("all"))
            # 使用新的变量名
        )

        self.sequence_control_canvas.create_window((0, 0), window=self.sequence_frame, anchor="nw")
        self.sequence_control_canvas.configure(yscrollcommand=seq_scrollbar.set)

        self.sequence_control_canvas.pack(side="left", fill="both", expand=True, pady=(10, 0))
        seq_scrollbar.pack(side="right", fill="y", pady=(10, 0))

        # 初始化序列控制
        self.update_sequence_controls()

    def create_control_buttons(self, parent):
        """创建控制按钮"""
        button_frame = ttk.LabelFrame(parent, text="Plot Controls", padding=10)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        # 更新按钮
        ttk.Button(button_frame, text="Update Basic Plots", command=self.update_plots,
                   style="Accent.TButton").pack(fill=tk.X, pady=2)

        # 图像控制按钮
        control_frame = ttk.Frame(button_frame)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(control_frame, text="Reset View", command=self.reset_view).pack(side=tk.TOP, fill=tk.X, pady=1)

        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, pady=2)
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.RIGHT, padx=2, fill=tk.X,
                                                                            expand=True)

        ttk.Button(control_frame, text="Pan Mode", command=self.enable_pan_mode).pack(side=tk.TOP, fill=tk.X, pady=1)

    def create_shape_position_controls(self, parent):
        """创建Shape位置查看控制"""
        section4 = ttk.LabelFrame(parent, text="Shape Position Comparison", padding=10)
        section4.pack(fill=tk.X, padx=5, pady=5)

        # 第一个对比组
        compare1_frame = ttk.LabelFrame(section4, text="Comparison 1", padding=5)
        compare1_frame.pack(fill=tk.X, pady=2)

        # Sample 1
        sample1_frame = ttk.Frame(compare1_frame)
        sample1_frame.pack(fill=tk.X)
        ttk.Label(sample1_frame, text="Instance:").pack(side=tk.LEFT)
        self.pos_sample1_var = tk.IntVar(value=1)
        self.pos_sample1_spinbox = ttk.Spinbox(sample1_frame, from_=1, to=1000, textvariable=self.pos_sample1_var,
                                               width=10)
        self.pos_sample1_spinbox.pack(side=tk.LEFT, padx=5)

        ttk.Label(sample1_frame, text="Shape:").pack(side=tk.LEFT, padx=(10, 0))
        self.pos_shape1_var = tk.IntVar(value=1)
        self.pos_shape1_spinbox = ttk.Spinbox(sample1_frame, from_=1, to=1000, textvariable=self.pos_shape1_var,
                                              width=10)
        self.pos_shape1_spinbox.pack(side=tk.LEFT, padx=5)

        # 第二个对比组
        compare2_frame = ttk.LabelFrame(section4, text="Comparison 2", padding=5)
        compare2_frame.pack(fill=tk.X, pady=2)

        # Sample 2
        sample2_frame = ttk.Frame(compare2_frame)
        sample2_frame.pack(fill=tk.X)
        ttk.Label(sample2_frame, text="Instance:").pack(side=tk.LEFT)
        self.pos_sample2_var = tk.IntVar(value=1)
        self.pos_sample2_spinbox = ttk.Spinbox(sample2_frame, from_=1, to=1000, textvariable=self.pos_sample2_var,
                                               width=10)
        self.pos_sample2_spinbox.pack(side=tk.LEFT, padx=5)

        ttk.Label(sample2_frame, text="Shape:").pack(side=tk.LEFT, padx=(10, 0))
        self.pos_shape2_var = tk.IntVar(value=1)
        self.pos_shape2_spinbox = ttk.Spinbox(sample2_frame, from_=1, to=1000, textvariable=self.pos_shape2_var,
                                              width=10)
        self.pos_shape2_spinbox.pack(side=tk.LEFT, padx=5)

        # 对比按钮
        ttk.Button(section4, text="Compare Two Positions", command=self.compare_shape_positions).pack(fill=tk.X,
                                                                                                      pady=10)

    def create_heatmap_controls(self, parent):
        """创建Heatmap控制区域"""
        heatmap_frame = ttk.LabelFrame(parent, text="Heatmap Controls", padding=10)
        heatmap_frame.pack(fill=tk.X, padx=5, pady=5)

        # 加载Heatmap NPY文件
        ttk.Button(heatmap_frame, text="Load Heatmap Data (.npy)",
                   command=self.load_heatmap_data).pack(fill=tk.X, pady=2)

        # Heatmap数据信息显示
        self.heatmap_info_label = ttk.Label(heatmap_frame, text="No heatmap data loaded",
                                            foreground="red")
        self.heatmap_info_label.pack(pady=2)

        # Sample序号选择
        sample_frame = ttk.Frame(heatmap_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sample_frame, text="Instance Number:").pack(side=tk.LEFT)
        self.heatmap_sample_var = tk.IntVar(value=1)
        self.heatmap_sample_spinbox = ttk.Spinbox(sample_frame, from_=1, to=1000,
                                                  textvariable=self.heatmap_sample_var, width=15)
        self.heatmap_sample_spinbox.pack(side=tk.RIGHT)

        # Shape范围选择
        range_frame = ttk.LabelFrame(heatmap_frame, text="Shape Number Range", padding=5)
        range_frame.pack(fill=tk.X, pady=5)

        start_frame = ttk.Frame(range_frame)
        start_frame.pack(fill=tk.X, pady=2)
        ttk.Label(start_frame, text="Start Shape:").pack(side=tk.LEFT)
        self.heatmap_start_var = tk.IntVar(value=1)
        self.heatmap_start_spinbox = ttk.Spinbox(start_frame, from_=1, to=1000,
                                                 textvariable=self.heatmap_start_var, width=15)
        self.heatmap_start_spinbox.pack(side=tk.RIGHT)

        end_frame = ttk.Frame(range_frame)
        end_frame.pack(fill=tk.X, pady=2)
        ttk.Label(end_frame, text="End Shape:").pack(side=tk.LEFT)
        self.heatmap_end_var = tk.IntVar(value=10)
        self.heatmap_end_spinbox = ttk.Spinbox(end_frame, from_=1, to=1000,
                                               textvariable=self.heatmap_end_var, width=15)
        self.heatmap_end_spinbox.pack(side=tk.RIGHT)

        # 更新按钮
        ttk.Button(heatmap_frame, text="Update Heatmap",
                   command=self.update_heatmap, style="Accent.TButton").pack(fill=tk.X, pady=10)

        # 功能说明
        info_frame = ttk.LabelFrame(heatmap_frame, text="Interactive Comparison", padding=5)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, text="• Click one point on heatmap to compare two shapes", 
                  font=("TkDefaultFont", 8)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="• X-axis = Shape 1, Y-axis = Shape 2", 
                  font=("TkDefaultFont", 8)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="• Uses current Instance Number for comparison",
                  font=("TkDefaultFont", 8)).pack(anchor=tk.W)
        ttk.Label(info_frame, text="• Results display in Basic Visualization tab", 
                  font=("TkDefaultFont", 8)).pack(anchor=tk.W)

        # 显示点击信息的标签
        self.click_info_label = ttk.Label(heatmap_frame, text="Click one point on heatmap to compare shapes",
                                          foreground="blue")
        self.click_info_label.pack(pady=2)

    def create_attention_controls(self, parent):
        """创建Attention控制区域"""
        attention_frame = ttk.LabelFrame(parent, text="Attention Controls", padding=10)
        attention_frame.pack(fill=tk.X, padx=5, pady=5)

        # 加载Attention文件
        ttk.Button(attention_frame, text="Load Attention Data (.npy)",
                   command=self.load_attention_data).pack(fill=tk.X, pady=2)

        # 数据信息显示
        self.attention_info_label = ttk.Label(attention_frame, text="No attention data loaded",
                                              foreground="red")
        self.attention_info_label.pack(pady=2)

        # Sample选择
        sample_frame = ttk.Frame(attention_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sample_frame, text="Instance Number:").pack(side=tk.LEFT)
        self.attention_sample_var = tk.IntVar(value=1)
        self.attention_sample_spinbox = ttk.Spinbox(sample_frame, from_=1, to=1000,
                                                    textvariable=self.attention_sample_var, width=15)
        self.attention_sample_spinbox.pack(side=tk.RIGHT)

        # Shape数量选择
        count_frame = ttk.Frame(attention_frame)
        count_frame.pack(fill=tk.X, pady=5)
        ttk.Label(count_frame, text="Number of Shapes:").pack(side=tk.LEFT)
        self.attention_count_var = tk.IntVar(value=10)
        self.attention_count_spinbox = ttk.Spinbox(count_frame, from_=1, to=1000,
                                                   textvariable=self.attention_count_var, width=15)
        self.attention_count_spinbox.pack(side=tk.RIGHT)

        # 更新按钮
        ttk.Button(attention_frame, text="Update Attention Plot",
                   command=self.update_attention_plot, style="Accent.TButton").pack(fill=tk.X, pady=10)

        # 下载索引按钮
        ttk.Button(attention_frame, text="Export Top Shapes Indices",
                   command=self.download_indices).pack(fill=tk.X, pady=2)

    def create_right_visualization(self, parent):
        """创建右侧可视化区域"""
        viz_frame = ttk.Frame(parent)
        parent.add(viz_frame, weight=3)

        # 创建标签页
        viz_notebook = ttk.Notebook(viz_frame)
        viz_notebook.pack(fill=tk.BOTH, expand=True)

        # 基础可视化标签页
        basic_viz_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(basic_viz_frame, text="Basic Visualization")
        self.create_basic_visualization(basic_viz_frame)

        # 高级可视化标签页
        advanced_viz_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(advanced_viz_frame, text="Advanced Analysis")
        self.create_advanced_visualization(advanced_viz_frame)

    def create_basic_visualization(self, parent):
        """创建基础可视化区域"""
        # 创建上下分割的可视化区域
        viz_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        viz_paned.pack(fill=tk.BOTH, expand=True)

        # 上半部分：时间序列可视化
        upper_frame = ttk.LabelFrame(viz_paned, text="Time Series Data Visualization")
        viz_paned.add(upper_frame, weight=2)

        # 下半部分：Shape位置比较
        lower_frame = ttk.LabelFrame(viz_paned, text="Shape Position Comparison")
        viz_paned.add(lower_frame, weight=1)

        # 创建matplotlib图形
        self.create_upper_plots(upper_frame)
        self.create_shape_comparison_plot(lower_frame)

    def create_advanced_visualization(self, parent):
        """创建高级可视化区域"""
        # 创建上下分割的可视化区域
        viz_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        viz_paned.pack(fill=tk.BOTH, expand=True)

        # 上半部分：Heatmap
        heatmap_frame = ttk.LabelFrame(viz_paned, text="Attention Weight Visualization")
        viz_paned.add(heatmap_frame, weight=1)

        # 下半部分：Attention
        lower_frame = ttk.LabelFrame(viz_paned, text="Attention Values Visualization")
        viz_paned.add(lower_frame, weight=1)

        # 创建matplotlib图形
        self.create_heatmap_plot(heatmap_frame)
        self.create_attention_plot(lower_frame)

    def create_upper_plots(self, parent):
        """创建上半部分的图形"""
        self.upper_fig = Figure(figsize=(14, 8), dpi=100)
        self.upper_canvas = FigureCanvasTkAgg(self.upper_fig, parent)
        self.upper_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        self.upper_toolbar = NavigationToolbar2Tk(self.upper_canvas, parent)
        self.upper_toolbar.update()

    def create_shape_comparison_plot(self, parent):
        """创建Shape位置比较图形"""
        self.shape_comparison_fig = Figure(figsize=(14, 6), dpi=100)
        self.shape_comparison_canvas = FigureCanvasTkAgg(self.shape_comparison_fig, parent)
        self.shape_comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        self.shape_comparison_toolbar = NavigationToolbar2Tk(self.shape_comparison_canvas, parent)
        self.shape_comparison_toolbar.update()

    def create_heatmap_plot(self, parent):
        """创建Heatmap图形"""
        self.heatmap_fig = Figure(figsize=(8, 6), dpi=100)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, parent)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        self.heatmap_toolbar = NavigationToolbar2Tk(self.heatmap_canvas, parent)
        self.heatmap_toolbar.update()

        # 绑定点击事件
        self.heatmap_canvas.mpl_connect('button_press_event', self.on_heatmap_click)

    def create_attention_plot(self, parent):
        """创建Attention图形"""
        self.attention_fig = Figure(figsize=(14, 6), dpi=100)
        self.attention_canvas = FigureCanvasTkAgg(self.attention_fig, parent)
        self.attention_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 添加工具栏
        self.attention_toolbar = NavigationToolbar2Tk(self.attention_canvas, parent)
        self.attention_toolbar.update()

        # 绑定鼠标事件
        self.attention_canvas.mpl_connect('motion_notify_event', self.on_attention_hover)
        self.attention_canvas.mpl_connect('axes_leave_event', self.on_attention_leave)

    # 文件操作方法
    def browse_npz_file(self):
        """浏览NPZ文件"""
        filename = filedialog.askopenfilename(
            title="Choose NPZ File",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )
        if filename:
            self.npz_path_var.set(filename)

    def browse_npy_file(self):
        """浏览NPY文件"""
        filename = filedialog.askopenfilename(
            title="Choose NPY File",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            self.npy_path_var.set(filename)

    def load_data(self):
        """加载数据文件"""
        try:
            npz_path = self.npz_path_var.get()
            npy_path = self.npy_path_var.get()

            if not npz_path or not npy_path:
                messagebox.showerror("Error", "Please choose both NPZ and NPY files first")
                return

            # 加载NPZ文件
            self.npz_data = np.load(npz_path)
            self.arr_0 = self.npz_data['arr_0']  # (sample, shape_number, shape_length)
            self.arr_1 = self.npz_data['arr_1']  # (sample, shape_number, VP)

            # 加载NPY文件
            self.x_train = np.load(npy_path)  # (sample, length, dimension_number)

            # 验证数据格式
            if len(self.arr_0.shape) != 3 or len(self.arr_1.shape) != 3:
                raise ValueError("NPZ file data format is incorrect!")
            if len(self.x_train.shape) != 3:
                raise ValueError("NPY file data format is incorrect!")

            # 更新控件范围
            self.update_control_ranges()

            # 更新信息显示
            info_text = f"Data loaded successfully!\n\n"
            info_text += f"NPZ File Information:\n"
            info_text += f"  arr_0 shape: {self.arr_0.shape}\n"
            info_text += f"  arr_1 shape: {self.arr_1.shape}\n\n"
            info_text += f"NPY File Information:\n"
            info_text += f"  x_train shape: {self.x_train.shape}\n\n"
            info_text += f"Data ranges:\n"
            info_text += f"  Instance Number: {self.x_train.shape[0]}\n"
            info_text += f"  Time Length: {self.x_train.shape[1]}\n"
            info_text += f"  Variable Number: {self.x_train.shape[2]}\n"
            info_text += f"  Shape Number: {self.arr_0.shape[1]}"

            self.data_info_text.config(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, info_text)
            self.data_info_text.config(state=tk.DISABLED)

            # 初始化图形
            self.update_plots()

            messagebox.showinfo("Success", "Data loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")
            self.data_info_text.config(state=tk.NORMAL)
            self.data_info_text.delete(1.0, tk.END)
            self.data_info_text.insert(tk.END, f"Loading failed: {str(e)}")
            self.data_info_text.config(state=tk.DISABLED)

    def load_heatmap_data(self):
        """加载Heatmap数据"""
        filename = filedialog.askopenfilename(
            title="Select Heatmap Data File",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.heatmap_data = np.load(filename)

                # 验证数据格式
                if len(self.heatmap_data.shape) != 3:
                    raise ValueError("Data should be 3D (instance, shape_number, shape_number)")
                if self.heatmap_data.shape[1] != self.heatmap_data.shape[2]:
                    raise ValueError("Second and third variables should be equal")

                # 更新控件范围
                sample_count = self.heatmap_data.shape[0]
                shape_count = self.heatmap_data.shape[1]

                self.heatmap_sample_spinbox.config(to=sample_count)
                self.heatmap_start_spinbox.config(to=shape_count)
                self.heatmap_end_spinbox.config(to=shape_count)
                self.heatmap_end_var.set(min(20, shape_count))

                # 更新信息显示
                info_text = f"Heatmap loaded: {self.heatmap_data.shape}"
                self.heatmap_info_label.config(text=info_text, foreground="green")

                messagebox.showinfo("Success", "Heatmap data loaded successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Error loading heatmap data: {str(e)}")
                self.heatmap_info_label.config(text="Load failed", foreground="red")

    def load_attention_data(self):
        """加载Attention数据"""
        filename = filedialog.askopenfilename(
            title="Select Attention Data File",
            filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.attention_data = np.load(filename)

                # 验证数据格式
                if len(self.attention_data.shape) != 3:
                    raise ValueError("Data should be 3D (instance_number, shape_number, value_number)")

                # 对每个sample进行排序并保留原始索引
                self.process_attention_data()

                # 更新控件范围
                sample_count = self.attention_data.shape[0]
                shape_count = self.attention_data.shape[1]

                self.attention_sample_spinbox.config(to=sample_count)
                self.attention_count_spinbox.config(to=shape_count)
                self.attention_count_var.set(min(15, shape_count))

                # 更新信息显示
                info_text = f"Attention loaded: {self.attention_data.shape}"
                self.attention_info_label.config(text=info_text, foreground="green")

                messagebox.showinfo("Success", "Attention data loaded successfully!")

            except Exception as e:
                messagebox.showerror("Error", f"Error loading attention data: {str(e)}")
                self.attention_info_label.config(text="Load failed", foreground="red")

    def process_attention_data(self):
        """处理attention数据，按sample排序并保留原始索引"""
        if self.attention_data is None:
            return

        sample_count, shape_count, value_count = self.attention_data.shape

        # 为每个sample创建排序数据和索引
        self.sorted_attention_data = np.zeros_like(self.attention_data)
        self.original_indices = np.zeros((sample_count, shape_count), dtype=int)
        self.all_samples_indices = []  # 存储所有sample的索引列表

        for sample_idx in range(sample_count):
            # 计算每个shape的平均attention值
            mean_values = np.mean(self.attention_data[sample_idx], axis=1)

            # 获取从大到小的排序索引
            sorted_indices = np.argsort(mean_values)[::-1]

            # 存储排序后的数据和原始索引
            self.sorted_attention_data[sample_idx] = self.attention_data[sample_idx][sorted_indices]
            self.original_indices[sample_idx] = sorted_indices

            # 添加到所有sample的索引列表
            self.all_samples_indices.append([sample_idx + 1, sorted_indices.tolist()])

    def update_control_ranges(self):
        """更新控件的范围"""
        if self.x_train is not None:
            sample_count = self.x_train.shape[0]
            length = self.x_train.shape[1]
            dimension_count = self.x_train.shape[2]

            # 更新位置查看控件
            self.pos_sample1_spinbox.config(to=sample_count)
            self.pos_sample2_spinbox.config(to=sample_count)

            # 更新序列控制控件
            for controls in self.sequence_controls:
                controls['instance_spinbox'].config(to=sample_count)
                controls['variable_spinbox'].config(to=dimension_count)
                controls['start_spinbox'].config(to=length - 1)
                controls['end_spinbox'].config(to=length)

        if self.arr_0 is not None:
            shape_number_count = self.arr_0.shape[1]
            # 更新shape相关控件
            self.pos_shape1_spinbox.config(to=shape_number_count)
            self.pos_shape2_spinbox.config(to=shape_number_count)

    def update_sequence_controls(self):
        """更新序列控制界面"""
        # 清除现有控件
        for widget in self.sequence_frame.winfo_children():
            widget.destroy()

        self.sequence_controls = []
        plot_count = self.plot_count_var.get()

        for i in range(plot_count):
            # 为每个序列创建控制组
            seq_frame = ttk.LabelFrame(self.sequence_frame, text=f"Sequence {i + 1} Setting", padding=5)
            seq_frame.pack(fill=tk.X, pady=2)

            # 创建控制变量字典
            controls = {}

            # Sample序号（从1开始）
            ttk.Label(seq_frame, text="Instance:").grid(row=0, column=0, sticky="w", padx=2)
            controls['instance'] = tk.IntVar(value=1)
            sample_spinbox = ttk.Spinbox(seq_frame, from_=1, to=1000, textvariable=controls['instance'], width=10)
            sample_spinbox.grid(row=0, column=1, padx=2, pady=1)

            # Dimension序号（从1开始）
            ttk.Label(seq_frame, text="Variable:").grid(row=0, column=2, sticky="w", padx=2)
            controls['variable'] = tk.IntVar(value=1)
            dim_spinbox = ttk.Spinbox(seq_frame, from_=1, to=1000, textvariable=controls['variable'], width=10)
            dim_spinbox.grid(row=0, column=3, padx=2, pady=1)

            # 初始时间
            ttk.Label(seq_frame, text="Start:").grid(row=1, column=0, sticky="w", padx=2)
            controls['start_time'] = tk.IntVar(value=0)
            start_spinbox = ttk.Spinbox(seq_frame, from_=0, to=999, textvariable=controls['start_time'], width=10)
            start_spinbox.grid(row=1, column=1, padx=2, pady=1)

            # 序列长度
            ttk.Label(seq_frame, text="End:").grid(row=1, column=2, sticky="w", padx=2)
            controls['end_time'] = tk.IntVar(value=100)
            end_spinbox = ttk.Spinbox(seq_frame, from_=1, to=999, textvariable=controls['end_time'], width=10)
            end_spinbox.grid(row=1, column=3, padx=2, pady=1)

            # 保存控件引用以便后续更新范围
            controls['instance_spinbox'] = sample_spinbox
            controls['variable_spinbox'] = dim_spinbox
            controls['start_spinbox'] = start_spinbox
            controls['end_spinbox'] = end_spinbox

            self.sequence_controls.append(controls)

    # 可视化更新方法
    def update_plots(self):
        """更新基础图形显示"""
        if self.x_train is None or self.arr_0 is None:
            messagebox.showwarning("Warning", "Please load data files first!")
            return

        try:
            self.update_upper_plots()
        except Exception as e:
            messagebox.showerror("Error", f"Error updating plots: {str(e)}")

    def update_upper_plots(self):
        """更新上半部分图形"""
        self.upper_fig.clear()

        plot_count = self.plot_count_var.get()

        # 根据图片数量确定子图布局
        if plot_count == 1:
            subplot_layout = (1, 1)
        elif plot_count == 2:
            subplot_layout = (1, 2)
        elif plot_count == 3:
            subplot_layout = (1, 3)
        elif plot_count == 4:
            subplot_layout = (2, 2)
        else:
            subplot_layout = (1, 1)

        for i in range(plot_count):
            if i < len(self.sequence_controls):
                controls = self.sequence_controls[i]

                sample_idx = controls['instance'].get() - 1  # 转换为0索引
                dimension_idx = controls['variable'].get() - 1  # 转换为0索引
                start_time = controls['start_time'].get()
                end_time = controls['end_time'].get()

                # 验证参数
                if sample_idx >= self.x_train.shape[0] or sample_idx < 0:
                    sample_idx = 0
                if dimension_idx >= self.x_train.shape[2] or dimension_idx < 0:
                    dimension_idx = 0

                # 验证时间范围
                max_time = self.x_train.shape[1]
                if start_time < 0:
                    start_time = 0
                if end_time > max_time:
                    end_time = max_time
                if start_time >= end_time:
                    end_time = min(start_time + 1, max_time)

                # 提取数据
                seq_length = end_time - start_time
                data_to_plot = self.x_train[sample_idx, start_time:end_time, dimension_idx]

                # 创建实际的时间轴（从start_time到end_time）
                time_axis = range(start_time, end_time)

                # 创建子图
                ax = self.upper_fig.add_subplot(subplot_layout[0], subplot_layout[1], i + 1)
                ax.plot(time_axis, data_to_plot, linewidth=2, label=f'Seq {i + 1}')
                ax.set_title(
                    f'Sequence {i + 1}: Instance {sample_idx + 1}, Variable {dimension_idx + 1}\nTime {start_time}-{end_time - 1} (Length: {seq_length})')
                ax.set_xlabel('Time Index')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.legend()

        self.upper_fig.tight_layout()
        self.upper_canvas.draw()

    def update_shape_comparison_plot(self):
        """更新Shape位置比较图形"""
        self.shape_comparison_fig.clear()

        ax = self.shape_comparison_fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'Use "Compare Two Positions" button\nto display shape comparison',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])

        self.shape_comparison_fig.tight_layout()
        self.shape_comparison_canvas.draw()

    def update_heatmap(self):
        """更新Heatmap显示"""
        if self.heatmap_data is None:
            messagebox.showwarning("Warning", "Please load heatmap data first!")
            return

        try:
            sample_idx = self.heatmap_sample_var.get() - 1  # 转换为0索引
            start_shape = self.heatmap_start_var.get() - 1  # 转换为0索引
            end_shape = self.heatmap_end_var.get()  # 保持为结束位置

            # 验证参数
            if sample_idx >= self.heatmap_data.shape[0] or sample_idx < 0:
                sample_idx = 0
            if start_shape < 0:
                start_shape = 0
            if end_shape > self.heatmap_data.shape[1]:
                end_shape = self.heatmap_data.shape[1]
            if start_shape >= end_shape:
                messagebox.showerror("Error", "Start shape must be less than end shape!")
                return

            # 清除之前的图形，但保留colorbar
            if self.current_heatmap_ax is not None:
                self.current_heatmap_ax.clear()
            else:
                self.heatmap_fig.clear()

            # 提取数据切片
            data_slice = self.heatmap_data[sample_idx, start_shape:end_shape, start_shape:end_shape]

            # 创建或更新heatmap
            if self.current_heatmap_ax is None:
                self.current_heatmap_ax = self.heatmap_fig.add_subplot(1, 1, 1)

            # 使用学术界专用的颜色（viridis或plasma）
            im = self.current_heatmap_ax.imshow(data_slice, cmap='viridis', aspect='auto', interpolation='nearest')

            # 设置标题
            self.current_heatmap_ax.set_title(f'Heatmap: Instance {sample_idx + 1}, Shapes {start_shape + 1}-{end_shape}')

            # 隐藏坐标轴数字
            self.current_heatmap_ax.set_xticks([])
            self.current_heatmap_ax.set_yticks([])
            self.current_heatmap_ax.set_xlabel('Shape Index')
            self.current_heatmap_ax.set_ylabel('Shape Index')

            # 存储当前显示的信息，用于点击事件
            self.current_start_shape = start_shape
            self.current_end_shape = end_shape
            self.current_sample_idx = sample_idx

            # 只在第一次或colorbar不存在时添加colorbar
            if self.heatmap_colorbar is None:
                self.heatmap_colorbar = self.heatmap_fig.colorbar(im, ax=self.current_heatmap_ax)
                self.heatmap_colorbar.set_label('Value')
            else:
                # 更新现有colorbar的映射
                self.heatmap_colorbar.mappable.set_array(data_slice)
                self.heatmap_colorbar.mappable.set_clim(vmin=data_slice.min(), vmax=data_slice.max())

            self.heatmap_fig.tight_layout()
            self.heatmap_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error updating heatmap: {str(e)}")

    def update_attention_plot(self):
        """更新Attention图表显示"""
        if self.attention_data is None or self.sorted_attention_data is None:
            messagebox.showwarning("Warning", "Please load attention data first!")
            return

        try:
            sample_idx = self.attention_sample_var.get() - 1  # 转换为0索引
            shape_count = self.attention_count_var.get()

            # 验证参数
            if sample_idx >= self.attention_data.shape[0] or sample_idx < 0:
                sample_idx = 0
            if shape_count > self.attention_data.shape[1]:
                shape_count = self.attention_data.shape[1]

            # 清除之前的图形和注释
            self.attention_fig.clear()
            self.attention_annotations = []

            # 获取排序后的数据和原始索引
            sorted_data = self.sorted_attention_data[sample_idx, :shape_count]
            original_idx = self.original_indices[sample_idx, :shape_count]

            # 计算平均值用于柱状图显示
            mean_values = np.mean(sorted_data, axis=1)

            # 创建柱状图
            self.attention_ax = self.attention_fig.add_subplot(1, 1, 1)

            self.attention_bars = self.attention_ax.bar(range(shape_count), mean_values,
                                                        color='steelblue', alpha=0.7)

            # 设置标题和标签
            self.attention_ax.set_title(
                f'Attention Values: Instance {sample_idx + 1}, Top {shape_count} Shapes (High to Low)')
            self.attention_ax.set_xlabel('Rank (High to Low)')
            self.attention_ax.set_ylabel('Attention Value')
            self.attention_ax.grid(True, alpha=0.3)

            # 隐藏x轴标签
            self.attention_ax.set_xticks([])

            # 存储原始索引用于hover显示
            self.current_attention_indices = original_idx

            self.attention_fig.tight_layout()
            self.attention_canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Error updating attention plot: {str(e)}")

    # 交互事件处理方法
    def on_heatmap_click(self, event):
        """处理heatmap点击事件"""
        if event.inaxes != self.current_heatmap_ax:
            return

        if self.heatmap_data is None or self.current_heatmap_ax is None:
            return

        try:
            # 获取点击的像素坐标
            x, y = int(event.xdata), int(event.ydata)

            # 转换为实际的shape坐标
            actual_x = x + self.current_start_shape  # 第一个shape number
            actual_y = y + self.current_start_shape  # 第二个shape number

            # 显示点击信息
            info_text = f"Clicked: Shape {actual_x + 1} vs Shape {actual_y + 1}. Generating comparison..."
            self.click_info_label.config(text=info_text, foreground="blue")

            # 直接生成shape position comparison
            self.generate_single_click_comparison(actual_x, actual_y)

        except (TypeError, IndexError):
            # 点击超出范围时忽略
            pass

    def generate_single_click_comparison(self, shape1_idx, shape2_idx):
        """基于单击生成shape position comparison"""
        if self.arr_1 is None or self.x_train is None:
            return

        try:
            # 获取当前sample（从heatmap控件获取）
            current_sample_idx = self.heatmap_sample_var.get() - 1

            # 验证索引范围
            if (current_sample_idx >= self.arr_1.shape[0] or current_sample_idx < 0 or
                shape1_idx >= self.arr_1.shape[1] or shape1_idx < 0 or
                shape2_idx >= self.arr_1.shape[1] or shape2_idx < 0):
                messagebox.showerror("Error", "Selected shapes are out of data range")
                return

            # 获取VP数据
            vp1_data = self.arr_1[current_sample_idx, shape1_idx, :]
            vp2_data = self.arr_1[current_sample_idx, shape2_idx, :]

            if len(vp1_data) < 4 or len(vp2_data) < 4:
                messagebox.showerror("Error", "VP data format is incorrect.")
                return

            # 解析VP数据
            length1, start1, end1, label1 = int(vp1_data[0]), int(vp1_data[1]), int(vp1_data[2]), vp1_data[3]
            length2, start2, end2, label2 = int(vp2_data[0]), int(vp2_data[1]), int(vp2_data[2]), vp2_data[3]

            # 调用现有的comparison显示方法
            self.show_comparison_window(
                current_sample_idx, shape1_idx, length1, start1, end1, label1,
                current_sample_idx, shape2_idx, length2, start2, end2, label2
            )

            # 更新信息显示
            info_text = f"Comparison generated: Instance {current_sample_idx + 1}, Shapes {shape1_idx + 1} vs {shape2_idx + 1}"
            self.click_info_label.config(text=info_text, foreground="green")

        except Exception as e:
            messagebox.showerror("Error", f"Error generating comparison: {str(e)}")
            self.click_info_label.config(text="Error generating comparison", foreground="red")

    def on_attention_hover(self, event):
        """处理attention plot的鼠标悬停事件"""
        if event.inaxes != getattr(self, 'attention_ax', None):
            return

        if not hasattr(self, 'attention_bars') or not hasattr(self, 'current_attention_indices'):
            return

        try:
            # 清除之前的注释
            for annotation in self.attention_annotations:
                annotation.remove()
            self.attention_annotations = []

            # 检查鼠标是否在某个柱子上
            for i, bar in enumerate(self.attention_bars):
                if bar.contains(event)[0]:
                    # 显示原始索引
                    original_idx = self.current_attention_indices[i]
                    height = bar.get_height()

                    annotation = self.attention_ax.annotate(
                        f'Original Idx: {original_idx}',
                        xy=(bar.get_x() + bar.get_width() / 2., height),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9
                    )
                    self.attention_annotations.append(annotation)
                    break

            self.attention_canvas.draw_idle()

        except Exception as e:
            pass  # 忽略hover错误

    def on_attention_leave(self, event):
        """处理鼠标离开attention plot事件"""
        try:
            # 清除所有注释
            for annotation in self.attention_annotations:
                annotation.remove()
            self.attention_annotations = []
            self.attention_canvas.draw_idle()
        except:
            pass

    def display_selected_sequences(self):
        """显示选中的两个shape的序列"""
        return  # 功能已删除

    # Shape位置比较方法
    def compare_shape_positions(self):
        """同时对比两个Shape位置"""
        if self.x_train is None or self.arr_0 is None or self.arr_1 is None:
            messagebox.showwarning("Warning", "Please load data files first.")
            return

        try:
            # 获取两组参数
            sample1_idx = self.pos_sample1_var.get() - 1
            shape1_idx = self.pos_shape1_var.get() - 1
            sample2_idx = self.pos_sample2_var.get() - 1
            shape2_idx = self.pos_shape2_var.get() - 1

            # 验证参数
            for idx, name in [(sample1_idx, "Instance1"), (sample2_idx, "Instance2")]:
                if idx >= self.arr_0.shape[0] or idx < 0:
                    messagebox.showerror("Error", f"{name} number is out of range")
                    return
            for idx, name in [(shape1_idx, "Shape1"), (shape2_idx, "Shape2")]:
                if idx >= self.arr_0.shape[1] or idx < 0:
                    messagebox.showerror("Error", f"{name} number is out of range")
                    return

            # 获取VP数据
            vp1_data = self.arr_1[sample1_idx, shape1_idx, :]
            vp2_data = self.arr_1[sample2_idx, shape2_idx, :]

            if len(vp1_data) < 4 or len(vp2_data) < 4:
                messagebox.showerror("Error", "VP data format is incorrect.")
                return

            # 解析VP数据
            length1, start1, end1, label1 = int(vp1_data[0]), int(vp1_data[1]), int(vp1_data[2]), vp1_data[3]
            length2, start2, end2, label2 = int(vp2_data[0]), int(vp2_data[1]), int(vp2_data[2]), vp2_data[3]

            # 创建对比窗口
            self.show_comparison_window(
                sample1_idx, shape1_idx, length1, start1, end1, label1,
                sample2_idx, shape2_idx, length2, start2, end2, label2
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error comparing Shape position: {str(e)}")

    def show_comparison_window(self, sample1_idx, shape1_idx, length1, start1, end1, label1,
                               sample2_idx, shape2_idx, length2, start2, end2, label2):
        """在主窗口的下方区域显示比较结果"""
        # 清除之前的图形
        self.shape_comparison_fig.clear()

        # 创建2个子图 (1行2列)
        ax1 = self.shape_comparison_fig.add_subplot(1, 2, 1)
        ax2 = self.shape_comparison_fig.add_subplot(1, 2, 2)

        # 绘制第一组时间序列数据
        if start1 < self.x_train.shape[1] and end1 <= self.x_train.shape[1]:
            ts1 = self.x_train[sample1_idx, :, 0]
            ax1.plot(ts1, linewidth=2, color='green', label='Time Series 1')
            if start1 < end1:
                ax1.axvspan(start1, end1, alpha=0.3, color='red',
                            label=f'Corresponding Shape')
                highlight_ts1 = ts1[start1:end1]
                ax1.plot(range(start1, end1), highlight_ts1, linewidth=3, color='red', alpha=0.8)
        ax1.set_title(
            f'Instance {sample1_idx + 1}, Shape {shape1_idx + 1}\nTime: {start1}-{end1}, Variable: {int(label1)}')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 绘制第二组时间序列数据
        if start2 < self.x_train.shape[1] and end2 <= self.x_train.shape[1]:
            ts2 = self.x_train[sample2_idx, :, 0]
            ax2.plot(ts2, linewidth=2, color='green', label='Time Series 2')
            if start2 < end2:
                ax2.axvspan(start2, end2, alpha=0.3, color='red',
                            label=f'Corresponding Shape')
                highlight_ts2 = ts2[start2:end2]
                ax2.plot(range(start2, end2), highlight_ts2, linewidth=3, color='red', alpha=0.8)
        ax2.set_title(
            f'Instance {sample2_idx + 1}, Shape {shape2_idx + 1}\nTime: {start2}-{end2}, Variable: {int(label2)}')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 添加总标题
        self.shape_comparison_fig.suptitle('Heatmap-Based Shape Comparison Analysis', fontsize=14)

        self.shape_comparison_fig.tight_layout()
        self.shape_comparison_canvas.draw()

    def download_indices(self):
        """下载所有sample的排序索引为NPY文件"""
        if not self.all_samples_indices:
            messagebox.showwarning("Warning", "No attention data loaded! Please load attention data first.")
            return

        try:
            # 选择保存位置
            filename = filedialog.asksaveasfilename(
                defaultextension=".npy",
                filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
            )

            if filename:
                # 保存为NPY格式
                np.save(filename, self.all_samples_indices)

                # 显示保存信息
                info_msg = f"Indices data saved successfully!\n\n"
                info_msg += f"Total instance: {len(self.all_samples_indices)}\n"
                info_msg += f"Format: [[instance1, [indices...]], [instance2, [indices...]], ...]\n"
                info_msg += f"File: {filename}\n\n"
                info_msg += "Example structure:\n"
                if self.all_samples_indices:
                    example = self.all_samples_indices[0]
                    info_msg += f"[{example[0]}, {example[1][:5]}...]"

                messagebox.showinfo("Success", info_msg)

        except Exception as e:
            messagebox.showerror("Error", f"Error saving indices: {str(e)}")

    # 视图控制方法
    def reset_view(self):
        """重置视图"""
        if hasattr(self, 'upper_toolbar'):
            self.upper_toolbar.home()
        if hasattr(self, 'shape_comparison_toolbar'):
            self.shape_comparison_toolbar.home()
        if hasattr(self, 'heatmap_toolbar'):
            self.heatmap_toolbar.home()
        if hasattr(self, 'attention_toolbar'):
            self.attention_toolbar.home()
        self.current_zoom = 1.0
        self.pan_offset = [0, 0]

    def zoom_in(self):
        """放大"""
        self.current_zoom *= 1.2
        if hasattr(self, 'upper_toolbar'):
            self.upper_toolbar.zoom()
        if hasattr(self, 'shape_comparison_toolbar'):
            self.shape_comparison_toolbar.zoom()

    def zoom_out(self):
        """缩小"""
        self.current_zoom /= 1.2
        if hasattr(self, 'upper_toolbar'):
            self.upper_toolbar.back()
        if hasattr(self, 'shape_comparison_toolbar'):
            self.shape_comparison_toolbar.back()

    def enable_pan_mode(self):
        """启用移动模式"""
        if hasattr(self, 'upper_toolbar'):
            self.upper_toolbar.pan()
        if hasattr(self, 'shape_comparison_toolbar'):
            self.shape_comparison_toolbar.pan()
        messagebox.showinfo("Pan Mode", "Pan mode enabled! Click and drag to move the plots!")


def main():
    root = tk.Tk()

    # 设置应用程序图标和样式
    try:
        # 尝试设置现代化的主题
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
    except:
        pass

    app = MergedVisualizationApp(root)

    # 设置窗口关闭事件
    def on_closing():
        if messagebox.askokcancel("Exit", "Are you sure you want to exit the application?"):
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # 启动主循环
    root.mainloop()


if __name__ == "__main__":
    main()