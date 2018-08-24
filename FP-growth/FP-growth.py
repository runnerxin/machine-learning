#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2018/8/23 15:31
# @Author  : runnerxin


class TreeNode:
    """
        Desc:
            构造FP-tree的结点数据结构
        Args:
            def add_times(self, times):         # 对count变量增加给定值
            def show(self, layer=1):            # 用于将树以文本形式显示
    """
    def __init__(self, name_value, occur_times, parent_node):
        self.name = name_value
        self.count = occur_times
        self.node_link = None           # 下一个同类的结点

        self.parent = parent_node
        self.children = {}

    def add_times(self, times):         # 对count变量增加给定值
        self.count += times

    def show(self, layer=1):            # 用于将树以文本形式显示
        print('  '*layer, self.name, '  ', self.count)
        for child in self.children.values():
            child.show(layer+1)


def load_simple_data():
    simple_data = [['r', 'z', 'h', 'j', 'p'],
                   ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                   ['z'],
                   ['r', 'x', 'n', 'o', 's'],
                   ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                   ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_data


def create_init_set(dataset):
    """
        Desc:
            统计每行数据出现的次数
        Args:
            dataset
        Returns:
            统计好的数据格式，{frozenset(line), times}
    """
    return_dict = {}
    for line in dataset:
        if frozenset(line) not in return_dict.keys():
            return_dict[frozenset(line)] = 1
        else:
            return_dict[frozenset(line)] += 1
    return return_dict


def update_header(header_table_point, target_node):
    """
        Desc:
            更新头指针，建立相同元素之间的关系，例如： 左边的r指向右边的r值，就是后出现的相同元素 指向 已经出现的元素
            从头指针的nodeLink开始，一直沿着nodeLink直到到达链表末尾。这就是链表。
        Args:
            header_table_point          满足minSup {所有的元素+(value, treeNode)}
            target_node                 Tree对象的子节点
        Returns:
            --
    """

    # 建立相同元素之间的关系，例如： 左边的r指向右边的r值
    while header_table_point.node_link is not None:
        header_table_point = header_table_point.node_link
    header_table_point.node_link = target_node


def update_tree(ordered_items, fp_tree, header_table, count):
    """
        Desc:
            针对每一行的数据，更新FP-tree
        Args:
            ordered_items       满足minSup 排序后的元素key的数组（大到小的排序
            fp_tree             空的Tree对象
            header_table        满足minSup {所有的元素+(value, treeNode)}
            count               原数据集中每一组Kay出现的次数
        Returns:
            --
    """
    if ordered_items[0] in fp_tree.children:        # 如果该元素在 inTree.children 这个字典中，就进行累加
        fp_tree.children[ordered_items[0]].add_times(count)
    else:                                           # 如果不存在子节点，我们为该inTree添加子节点
        fp_tree.children[ordered_items[0]] = TreeNode(ordered_items[0], count, fp_tree)

        if header_table[ordered_items[0]][1] is None:       # headerTable只记录第一次节点出现的位置
            header_table[ordered_items[0]][1] = fp_tree.children[ordered_items[0]]
        else:
            # 本质上是修改headerTable的key对应的Tree，的nodeLink值
            update_header(header_table[ordered_items[0]][1], fp_tree.children[ordered_items[0]])

    if len(ordered_items) > 1:
        # 递归的调用，count只要循环的进行累计加和而已，统计出节点的最后的统计值。
        update_tree(ordered_items[1:], fp_tree.children[ordered_items[0]], header_table, count)


def create_tree(init_set, min_support=1):
    """
        Desc:
            生成FP-tree
        Args:
            init_set            dist{行：出现次数}的样本数据
            min_support         最小的支持度
        Returns:
            fp_tree             FP-tree
            header_table        满足min_support {所有的元素+(value, treeNode)}
    """

    header_table = {}        # 支持度>=minSup的dist{所有元素：出现的次数}，用来记录字母第一次出现的结点
    for line in init_set:
        for item in line:
            if item not in header_table.keys():
                header_table[item] = init_set[line]
            else:
                header_table[item] += init_set[line]

    # 删除 headerTable中，元素次数<最小支持度的元素
    for k in list(header_table.keys()):  # python3中.keys()返回的是迭代器不是list,不能在遍历时对其改变。
        if header_table[k] < min_support:
            del (header_table[k])

    # print(header_table)
    frequency_item_set = set(header_table.keys())
    if len(frequency_item_set) == 0:                    # 如果不存在，直接返回None
        return None, None
    for k in header_table:
        header_table[k] = [header_table[k], None]       # 格式化： dist{元素key: [元素次数, None]}

    fp_tree = TreeNode('Null set', 1, None)
    for line, count in init_set.items():
        local = {}
        for item in line:
            if item in frequency_item_set:
                local[item] = header_table[item][0]     # header_table => dist{元素key: [元素次数, None]}

        # print('line=', line, '  local=', local)
        if len(local) > 0:
            sort_local = sorted(local.items(), key=lambda p: p[1], reverse=True)    # 从大到小的排序
            ordered_items = [v[0] for v in sort_local]
            # print(ordered_items)

            # 填充树，通过有序的orderedItems的第一位，进行顺序填充 第一层的子节点。
            update_tree(ordered_items, fp_tree, header_table, count)
    return fp_tree, header_table


def ascend_tree(leaf_node, prefix_path):
    """
        Desc:
            如果存在父节点，就记录当前节点的name值
        Args:
            leaf_node            查询的节点对于的nodeTree
            prefix_path          要查询的节点值
    """
    if leaf_node.parent is not None:
        prefix_path.append(leaf_node.name)
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_pat, tree_node):
    """
        Desc:
            基础数据集
        Args:
            basePat         要查询的节点值
            treeNode        查询的节点所在的当前nodeTree
        Returns:
            condPats        对非basePat的倒叙值作为key,赋值为count数
    """

    count_pats = {}

    while tree_node is not None:
        prefix_path = []
        ascend_tree(tree_node, prefix_path)     # 寻找改节点的父节点，相当于找到了该节点的频繁项集

        if len(prefix_path) > 1:
            count_pats[frozenset(prefix_path[1:])] = tree_node.count
        tree_node = tree_node.node_link

    return count_pats


def mine_tree(fp_tree, header_table, min_support, prefix_path, frequency_item):
    """
        Desc:
            创建条件FP树
        Args:
            fp_tree             myFPtree
            header_table        满足minSup {所有的元素+(value, treeNode)}
            min_support         最小支持项集
            prefix_path         preFix为newFreqSet上一次的存储记录，一旦没有myHead，就不会更新
            frequency_item      用来存储频繁子项的列表
        Returns:
            __
    """

    # 通过value进行从小到大的排序， 得到频繁项集的key
    frequency_set_key = [v[0] for v in sorted(header_table.items(), key=lambda p: p[1][0])]

    for base_pat in frequency_set_key:
        # prefix_path为new_frequency_set上一次的存储记录，一旦没有my_head，就不会更新
        new_frequency_set = prefix_path.copy()
        new_frequency_set.add(base_pat)

        frequency_item.append(new_frequency_set)
        cond_pat_base = find_prefix_path(base_pat, header_table[base_pat][1])

        # 构建FP-tree
        my_cond_tree, my_head = create_tree(cond_pat_base, min_support)

        # 挖掘条件 FP-tree, 如果myHead不为空，表示满足minSup {所有的元素+(value, treeNode)}
        if my_head is not None:
            my_cond_tree.show()
            # 递归 myHead 找出频繁项集
            mine_tree(my_cond_tree, my_head, min_support, new_frequency_set, frequency_item)


def test_fp_growth():
    # load样本数据
    simple_data = load_simple_data()

    # frozen set 格式化 并 重新装载 样本数据，对所有的行进行统计求和，格式: {行：出现次数}
    init_set = create_init_set(simple_data)

    # 创建FP树
    # 输入：dist{行：出现次数}的样本数据  和  最小的支持度
    # 输出：最终的PF-tree，通过循环获取第一层的节点，然后每一层的节点进行递归的获取每一行的字节点，也就是分支。
    #       然后所谓的指针，就是后来的指向已存在的
    my_fp_tree, my_header_tab = create_tree(init_set, 3)
    my_fp_tree.show()

    # 抽取条件模式基
    # 查询树节点的，频繁子项
    print('x --->', find_prefix_path('x', my_header_tab['x'][1]))
    print('z --->', find_prefix_path('z', my_header_tab['z'][1]))
    print('r --->', find_prefix_path('r', my_header_tab['r'][1]))

    # 创建条件模式基
    frequency_item_list = []
    mine_tree(my_fp_tree, my_header_tab, 3, set([]), frequency_item_list)
    print(frequency_item_list)


def test_on_read():

    # 新闻网站点击流中挖掘，例如：文章1阅读过的人，还阅读过什么？
    parse_data = [line.split() for line in open('input/kosarak.dat').readlines()]
    init_set = create_init_set(parse_data)
    my_fp_tree, my_header_tab = create_tree(init_set, 100000)

    my_frequency_item_list = []
    mine_tree(my_fp_tree, my_header_tab, 100000, set([]), my_frequency_item_list)

    print(my_frequency_item_list)


if __name__ == '__main__':

    # # 将树以文本形式显示
    # rootNode = TreeNode('pyramid', 9, 'None')
    # rootNode.children['eye'] = TreeNode('eye', 13, None)
    # rootNode.children['phoenix'] = TreeNode('phoenix', 3, None)
    # rootNode.show()

    # # 测试样本数据
    # test_fp_growth()

    # 项目实战测试
    test_on_read()




