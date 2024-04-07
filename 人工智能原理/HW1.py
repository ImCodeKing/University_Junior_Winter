class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def print_tree(node, indent=""):
    print(indent + node.value)
    for child in node.children:
        print_tree(child, indent + "  ")


def create_search_tree(max_depth):
    root = TreeNode("")
    node_a = TreeNode("A")
    node_b = TreeNode("B")
    node_c = TreeNode("C")

    root.children = [node_a, node_b, node_c]

    for node in [node_a, node_b, node_c]:
        node.children = [TreeNode(node.value + letter) for letter in ["A", "B", "C"]]
        if max_depth > 1:
            for child in node.children:
                child.children = [TreeNode(child.value + letter) for letter in ["A", "B", "C"]]
                if max_depth > 2:
                    for grandchild in child.children:
                        grandchild.children = [TreeNode(grandchild.value + letter) for letter in ["A", "B", "C"]]
                        if max_depth > 3:
                            for great_grandchild in grandchild.children:
                                great_grandchild.children = [TreeNode(great_grandchild.value + letter) for letter in
                                                             ["A", "B", "C"]]
                                if max_depth > 4:
                                    for great_great_grandchild in great_grandchild.children:
                                        great_great_grandchild.children = [
                                            TreeNode(great_great_grandchild.value + letter) for letter in
                                            ["A", "B", "C"]]
                                        if max_depth > 5:
                                            for great_great_great_grandchild in great_great_grandchild.children:
                                                great_great_great_grandchild.children = [
                                                    TreeNode(great_great_great_grandchild.value + letter) for letter in
                                                    ["A", "B", "C"]]
                                                if max_depth > 6:
                                                    for great_great_great_great_grandchild in great_great_great_grandchild.children:
                                                        great_great_great_great_grandchild.children = [
                                                            TreeNode(great_great_great_great_grandchild.value + letter)
                                                            for letter in ["A", "B", "C"]]
                                                        if max_depth > 7:
                                                            for great_great_great_great_great_grandchild in great_great_great_great_grandchild.children:
                                                                great_great_great_great_great_grandchild.children = [
                                                                    TreeNode(
                                                                        great_great_great_great_great_grandchild.value + letter)
                                                                    for letter in ["A", "B", "C"]]

    return root


if __name__ == "__main__":
    max_depth = 8
    tree_root = create_search_tree(max_depth)
    print("搜索树:")
    print_tree(tree_root)
