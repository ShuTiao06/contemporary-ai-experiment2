# 当代人工智能实验二：搜索算法

## 运行

需要 Python 3.10 或更高版本，只使用 Python 标准库。进入本目录后运行：

```bash
python experiment2.py
```

程序会在本目录生成 `experiment2_results.json` 和 `distance_matrix.csv`，并打印完整实验结果。代码固定按 PDF 样例运行；其他网格可直接调用下方函数。

## 实现接口

```python
from experiment2 import (
    astar, maze_shortest_path, weighted_path_cost,
    TreeNode, solve_minimax, solve_alpha_beta,
)

path = maze_shortest_path(start, goal, grid)  # 无解返回 []
cost = weighted_path_cost(start, goal, weighted_grid)  # 无解返回 -1

root = TreeNode(children=[TreeNode(value=3), TreeNode(value=5)])
value1 = solve_minimax(root)
value2 = solve_alpha_beta(root)
```

`astar(grid, start, goal, cost_mode, heuristic)` 还会返回路径、最小代价和搜索计数，便于比较 `heuristic="zero"` 与 `heuristic="manhattan"`。网格坐标均为 `(row, col)`；四邻接顺序为上、右、下、左。

## 算法约定

- 普通迷宫：进入每个可走格代价为 1，`X` 不可通行，起终点均包含在返回路径中。
- 加权地图：移动到相邻格的代价等于进入该格的权值；按题意进入 `E` 的代价为 0。
- 加权地图启发值使用 `max(0, Manhattan距离-1) × 最小普通格代价`。这样将免费进入终点的最后一步排除，仍保持可采纳。
- 博弈树从 MAX 开始，子节点按左到右访问。综合题效用为 `Score_A - Score_B`，每个宝藏收益为“价值 - 当前玩家到宝藏的最小移动代价”。搜索深度表示从当前状态起的行动层数。
- 综合题的 `greedy` 顺序按当前玩家的即时收益从高到低访问；在 MIN 层这等价于优先尝试对 A 更不利的宝藏，以改善 Alpha-Beta 剪枝。

## 输出文件

- `experiment2_results.json`：所有样例结果、搜索节点/叶节点/剪枝数和距离矩阵。
- `distance_matrix.csv`：A、B、T1–T4 之间的最小移动代价矩阵。
- `实验二报告.pdf`：算法说明、对照结果、分析与结论。
