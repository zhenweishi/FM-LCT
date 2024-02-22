import argparse
import anytree
from pathlib import Path

def pytree(path, max_level=1, max_show=3):
    path = Path(path)
    def dfs(cur_dir, parent_node, cur_level=0):
        if cur_level >= max_level:
            return
        to_show = sorted(cur_dir.iterdir())
        for i, child_path in enumerate(to_show):
            if i == max_show - 1 and max_show < len(to_show):
                anytree.Node(f"...", parent=parent_node)
            if child_path.is_dir():
                child_node = anytree.Node(child_path.name, parent=parent_node)
                dfs(child_path, child_node, cur_level+1)
            else:
                anytree.Node(child_path.name, parent=parent_node)  
            if i == max_show - 1 and max_show < len(to_show):
                break      

    root = anytree.Node(path.name)
    dfs(path, root, 0)
    return root

def tree(path, max_level=1, max_show=3):
    root_node = pytree(path, max_level, max_show)
    root_node = pytree(path, max_level, max_show)

    for pre, fill, node in anytree.RenderTree(root_node):
        print("%s%s" % (pre, node.name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a tree structure of a directory")
    parser.add_argument("path", help="The root directory path")
    parser.add_argument("--max_level", type=int, default=1, help="The maximum depth level to show in the tree")
    parser.add_argument("--max_show", type=int, default=3, help="The maximum number of child items to show for each directory")
    args = parser.parse_args()

    path = Path(args.path)
    
    if not path.exists() or not path.is_dir():
        print(f"{args.path} is not a valid directory path.")
        exit(1)

    root_node = pytree(path, args.max_level, args.max_show)

    for pre, fill, node in anytree.RenderTree(root_node):
        print("%s%s" % (pre, node.name))
