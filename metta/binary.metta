; Define a node with a value and two children
(: node (-> $value (BinaryTree $t) (BinaryTree $t) (BinaryTree $t)))

; Define an empty node (leaf)
(: nil (BinaryTree $t))


; 2. Constructing a Binary Tree:
; Let's construct the following binary tree:

; Define leaf nodes
(= leaf4 (node 4 nil nil))
(= leaf5 (node 5 nil nil))
(= leaf3 (node 3 nil nil))

; Define internal nodes
(= node2 (node 2 leaf4 leaf5))

; Define the root node
(= root (node 1 node2 leaf3))

(: check-tree (-> (BinaryTree $t) Bool))
(= (check-tree nil) True)
(= (check-tree (node $value $left $right))
   (and (check-tree $left) (check-tree $right)))

! (check-tree root)

; In-order Traversal
(: inorder-traverse (-> (BinaryTree $t) (List $t)))

; In-order traversal of a binary tree
(= (inorder-traverse nil) Nil)
(= (inorder-traverse (node $vlaue $left $right))
    (append (inorder-traverse $left) (Cons $value (inorder-traverse $right))))
(: in-order (-> nil))

! (inorder root)


