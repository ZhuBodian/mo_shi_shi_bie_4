import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

'''
这个第一开始是为了画决策树，具体实现例子可以参看统计学习方法decision_tree_test
'''


# 节点
class Node:
    def __init__(self, val=None, reason=None):
        self.val = val  # 节点值
        self.child = []  # 子节点列表
        self.depth = None
        self.parent_val = None
        self.reason = reason

    # 添加子节点,node:Tree.Node，注意这里不是str
    def add_child(self, node):
        self.child.append(node)


class TreeGenerate:

    # 假设每个节点的val都是独一无二的
    def __init__(self, val, node_type=Node, **kwargs):
        # node_type为节点类，要改写代码的话，往往继承改写节点的类就好了
        # val:str
        self.root = node_type(val=val, **kwargs)
        self.root.depth = 1
        self.root.parent_val = None

        self.hierarchy = [val]  # 以深度划分的多维list
        self.parent = [[None]]  # 父节点list，与hierarchy一一对应
        self.reason = [[None]]
        # self.tree = self.root  # 指向根节点
        self.depth = 1

    # 为指定父节点parent:str添加子节点child:str or str_list，及其原因reasons_str
    def add_child(self, parent_str, childs_str, reasons_str=None, node_type=Node, **kwargs):
        parent_node = Find(self.root).preorder(parent_str)  # 找到的父节点，且注意self.tree是根节点，是node类型，不是tree
        if reasons_str is None:
            reasons_str = len(childs_str) * [None]
        for i, child in enumerate(childs_str):  # 对当前输入进行枚举
            child_node = node_type(val=child, reason=reasons_str[i], **kwargs, sub_i=i)
            child_node.depth = parent_node.depth + 1
            child_node.parent_val = parent_str

            parent_node.add_child(child_node)

            # 更新树的深度
            if child_node.depth > self.depth:
                self.depth = child_node.depth

            # 更新层级结构与parent
            if len(self.hierarchy) < child_node.depth:  # 层级不够，添加新层
                temp = [child]
                self.hierarchy.append(temp)
                temp = [parent_str]
                self.parent.append(temp)
                temp = [reasons_str[i]]
                self.reason.append(temp)
            else:  # 在之前的老层上添加
                self.hierarchy[child_node.depth - 1].append(child)
                self.parent[child_node.depth - 1].append(parent_str)
                self.reason[child_node.depth - 1].append(reasons_str[i])

# 遍历
class Traverse:
    def __init__(self, tn):
        self.res = ''
        self.tn = tn  # tn相当于子树

    def preorder(self):
        if not self.tn:
            return
        self.res += self.tn.val
        for i in self.tn.child:
            self.res += Traverse(i).preorder()

        return self.res

# 在指定tree中寻找指定node_val，并返回该节点
class Find:
    def __init__(self, tn):
        self.tn = tn  # 是Node型变量，不是tree（所以应该传入的是一个tree变量的root属性）
        self.target_node = None

    def preorder(self, target_val):
        # 如果树中不存在target_val，返回None
        if self.tn.val == target_val:
            return self.tn
        for i in self.tn.child:
            self.target_node = Find(i).preorder(target_val)
            if self.target_node is None:  # 到达叶节点
                continue
            else:
                if self.target_node.val == target_val:
                    break

        return self.target_node

    # 返回树中的指定属性，list of something
    # att:str
    def return_nodes_att(self, att):
        return 0


# 画出多叉树
class TreePlot:
    def __init__(self, tree):
        self.tree = tree
        self.decision_node = dict(boxstyle="sawtooth", fc="0.8")  # 决策节点的属性。boxstyle为文本框的类型，sawtooth是锯齿形，
        self.line_text = dict(fc="0.8")
        # fc是边框线粗细,可以写为decisionNode={boxstyle:'sawtooth',fc:'0.8'}
        self.leaf_node = dict(boxstyle="round4", fc="0.8")  # 决策树叶子节点的属性
        self.arrow_args = dict(arrowstyle="<-")  # 箭头的属性

    def _plot_node(self, ax, node_text, parent_arrow_pt, text_center, reason):
        if parent_arrow_pt is None:  # 根节点单独处理
            ax.text(x=text_center[0], y=text_center[1], s=node_text, ha="center", va="center",
                    bbox=self.decision_node, fontsize=15)
        else:
            temp_x = (parent_arrow_pt[0] + text_center[0]) / 2  # 线上添加文本
            temp_y = (parent_arrow_pt[1] + text_center[1]) / 2
            ax.text(x=temp_x, y=temp_y, s=reason, ha="center", va="center", fontsize=10)
            ax.annotate(text=node_text, xy=parent_arrow_pt, xytext=text_center, va="center", ha="center",
                        bbox=self.decision_node, arrowprops=self.arrow_args, fontsize=15)

    def create_plot(self):
        # 输出中文
        mpl.rcParams['font.sans-serif'] = ["SimHei"]
        mpl.rcParams["axes.unicode_minus"] = False
        plt.figure()
        ax = plt.gca()

        node_position, parent_arrow_position, ax_lim = TreePlot(tree=self.tree)._position_cal()

        for depth_i in range(self.tree.depth):
            for j, element in enumerate(self.tree.hierarchy[depth_i]):
                node_text = element
                parent_arrow_pt = parent_arrow_position[depth_i][j]
                text_center = node_position[depth_i][j]
                TreePlot(self.tree)._plot_node(ax=ax, node_text=node_text, parent_arrow_pt=parent_arrow_pt,
                                               text_center=text_center, reason=self.tree.reason[depth_i][j])

        plt.xlim(left=ax_lim[0], right=ax_lim[2])
        plt.ylim(bottom=ax_lim[1], top=(0 - ax_lim[1]) / self.tree.depth * 0.5)
        plt.show()

    # 计算用于画图的几个节点数据
    def _position_cal(self):
        high_step = 20  # 每个节点上下间隔
        wide_step = 30  # 每个节点的左右单元
        node_position = [[(0, 0)]]
        ax_lim = [0, 0, 0]  # 所画图像的左，下，右的刻度
        parent_arrow_position = [[None]]
        max_hierarchy_i_nums = 0
        parent_list = copy.deepcopy(self.tree.parent)  # 浅copy都不行,因为是多维列表？
        reason_list = copy.deepcopy(self.tree.reason)
        position_dict = {self.tree.root.val: node_position[0][0]}  # 记录节点名：节点位置的字典

        # 校正self.tree.hierarchy的内部元素顺序，防止之后画图出现线的交叉
        for depth_i in range(1, self.tree.depth):
            hierarchy_i_nums = len(self.tree.hierarchy[depth_i])  # 当前层有几个节点
            hierarchy_i_1_nums = len(self.tree.hierarchy[depth_i - 1])  # 上一有几个节

            temp_dict = {}  # 记录上一层元素顺序
            temp = np.array([])  # 将当前层的父节点的字符，映射为父节点的顺序
            for j in range(hierarchy_i_1_nums):
                temp_dict[self.tree.hierarchy[depth_i - 1][j]] = j
            for j in range(hierarchy_i_nums):
                temp = np.append(temp, temp_dict[parent_list[depth_i][j]])
            idx = temp.argsort()  # temp排序，并返回原array的元素，在新array中的位置
            temp_list = []
            for j in range(hierarchy_i_nums):  # 排序通过映射，对改层元素进行排序
                temp_list.append(self.tree.hierarchy[depth_i][idx[j]])
                self.tree.parent[depth_i][j] = parent_list[depth_i][idx[j]]
                self.tree.reason[depth_i][j] = reason_list[depth_i][idx[j]]

            self.tree.hierarchy[depth_i] = temp_list
        parent_list = self.tree.parent.copy()  # 排序之后，更新一下这个变量

        # 计算node_position
        for depth_i in range(1, self.tree.depth):
            hierarchy_i_nums = len(self.tree.hierarchy[depth_i])  # 当前层有几个节点
            max_hierarchy_i_nums = (
                hierarchy_i_nums if hierarchy_i_nums > max_hierarchy_i_nums else max_hierarchy_i_nums)  # 记录最大宽度
            high_i = 0 - depth_i * high_step  # 当前层数的纵坐标
            temp_list = []
            middle = int(hierarchy_i_nums / 2)

            if hierarchy_i_nums % 2 == 0:  # 当前层有偶数个节点
                for i in range(middle):  # 左半边
                    temp_list.append((-(middle - i) * wide_step, high_i))

                for i in range(middle):  # 右半边
                    temp_list.append(((i + 1) * wide_step, high_i))

            else:  # 当前有奇数个节点
                for i in range(middle):  # 左半边
                    temp_list.append((-(middle - i) * wide_step, high_i))

                temp_list.append((0, high_i))  # 正中间

                for i in range(middle):  # 右半边
                    temp_list.append(((i + 1) * wide_step, high_i))
            node_position.append(temp_list)

        # 计算position_dict
        for depth_i in range(1, self.tree.depth):
            hierarchy_i_nums = len(self.tree.hierarchy[depth_i])
            for j in range(hierarchy_i_nums):
                position_dict[self.tree.hierarchy[depth_i][j]] = node_position[depth_i][j]

        # 计算parent_arrow_position
        for depth_i in range(1, self.tree.depth):
            hierarchy_i_nums = len(self.tree.hierarchy[depth_i])
            temp_list = []
            for j in range(hierarchy_i_nums):
                parent_val = parent_list[depth_i][j]
                temp_list.append(position_dict[parent_val])
            parent_arrow_position.append(temp_list)

        ax_lim[0] = -int(max_hierarchy_i_nums / 2) * wide_step - wide_step
        ax_lim[1] = high_i - high_step
        ax_lim[2] = -ax_lim[0]
        return node_position, parent_arrow_position, ax_lim
