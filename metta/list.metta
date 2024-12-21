; Here’s the implementation of the list operations in MeTTa, each with an example:

; 1. append
; Combines two lists.

(: Nil (List $t))
(: Cons (-> $t (List $t) (List $t)))
(: append (-> (List $t) (List $t) (List $t)))

(= (append Nil $list2) $list2)
(= (append (Cons $head $tail) $list2)
   (Cons $head (append $tail $list2)))

! (append (Cons 1 (Cons 2 Nil)) (Cons 3 (Cons 4 Nil)))

; 2. length
; Counts the number of elements in a list

(: length (-> (List $t) Number))
(= (length Nil) 0)
(= (length (Cons $head $tail)) (+ 1 (length $tail)))

! (length (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [3]


; 3. is-member
; Checks if a value is in the list.

(: is-member (-> $t (List $t) Bool))
(= (is-member $x Nil) False)
(= (is-member $x (Cons $head $tail))
   (or (== $x $head) (is-member $x $tail)))

! (is-member 2 (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [True]
! (is-member 4 (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [False]


; 4. max-value
; Finds the maximum value in a list.

(: max-value (-> (List Number) Number))
(= (max-value (Cons $head Nil)) $head)
(= (max-value (Cons $head $tail))
   (if (> $head (max-value $tail)) $head (max-value $tail)))


; ! (max-value (Cons 1 (Cons 3 (Cons 2 Nil)))) ; [3]

; 5. min-value
; Finds the minimum value in a list.

(: min-value (-> (List Number) Number))
(= (min-value (Cons $head Nil)) $head)
(= (min-value (Cons $head $tail))
   (if (< $head (min-value $tail)) $head (min-value $tail)))

; ! (min-value (Cons 1 (Cons 3 (Cons 2 Nil)))) ; [1]

; 6. push
; Adds an element to the front of the list.

(: push (-> $t (List $t) (List $t)))
(= (push $element $list) (Cons $element $list))

! (push 0 (Cons 1 (Cons 2 Nil))) ; [Cons 0 (Cons 1 (Cons 2 Nil))]

; 7. pop
; Removes the first element of the list.

(: pop (-> (List $t) (List $t)))
(= (pop (Cons $head $tail)) $tail)
(= (pop Nil) Nil) ; Handle empty list

! (pop (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [Cons 2 (Cons 3 Nil)]

; 8. remove-element
; Removes all occurrences of a specific element from the list.

(: remove-element (-> $t (List $t) (List $t)))
(= (remove-element $x Nil) Nil)
(= (remove-element $x (Cons $head $tail))
   (if (== $x $head) (remove-element $x $tail) (Cons $head (remove-element $x $tail))))

! (remove-element 2 (Cons 1 (Cons 2 (Cons 2 (Cons 3 Nil))))) ; [Cons 1 (Cons 3 Nil)]

; 9. remove-duplicate
; Removes duplicate elements from a list.

(: remove-duplicate (-> (List $t) (List $t)))
(= (remove-duplicate Nil) Nil)
(= (remove-duplicate (Cons $head $tail))
   (if (is-member $head $tail)
       (remove-duplicate $tail)
       (Cons $head (remove-duplicate $tail))))

! (remove-duplicate (Cons 1 (Cons 2 (Cons 1 (Cons 3 Nil))))) ; [Cons 1 (Cons 2 (Cons 3 Nil))]

; 10. map
; Applies a function to each element in the list.

(: map (-> (-> $t $u) (List $t) (List $u)))
(= (map $f Nil) Nil)
(= (map $f (Cons $head $tail))
   (Cons ($f $head) (map $f $tail)))

(: double (-> Number Number))
(= (double $x) (* 2 $x))

! (map double (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [Cons 2 (Cons 4 (Cons 6 Nil))]

; 11. filter
; Filters elements based on a predicate.

(: filter (-> (-> $t Bool) (List $t) (List $t)))
(= (filter $f Nil) Nil)
(= (filter $f (Cons $head $tail))
   (if ($f $head)
       (Cons $head (filter $f $tail))
       (filter $f $tail)))

(: is-even (-> Number Bool))
(= (is-even $x) (== (mod $x 2) 0))

! (filter is-even (Cons 1 (Cons 2 (Cons 3 (Cons 4 Nil))))) ; [Cons 2 (Cons 4 Nil)]

; 12. foldl
; Reduces a list from left to right using a function.

(: foldl (-> (-> $u $t $u) $u (List $t) $u))
(= (foldl $f $acc Nil) $acc)
(= (foldl $f $acc (Cons $head $tail))
   (foldl $f ($f $acc $head) $tail))

(: add (-> Number Number Number))
(= (add $x $y) (+ $x $y))

! (foldl add 0 (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [6]

; 13. foldr
; Reduces a list from right to left using a function.

(: foldr (-> (-> $t $u $u) $u (List $t) $u))
(= (foldr $f $acc Nil) $acc)
(= (foldr $f $acc (Cons $head $tail))
   ($f $head (foldr $f $acc $tail)))

(: add (-> Number Number Number))
(= (add $x $y) (+ $x $y))

! (foldr add 0 (Cons 1 (Cons 4 (Cons 3 Nil)))) ; [8]

; 14. reverse
; Reverses the elements in a list.

(: reverse (-> (List $t) (List $t)))
(: reverse-helper (-> (List $t) (List $t) (List $t)))

(= (reverse $list) (reverse-helper $list Nil))
(= (reverse-helper Nil $acc) $acc)
(= (reverse-helper (Cons $head $tail) $acc)
   (reverse-helper $tail (Cons $head $acc)))

! (reverse (Cons 1 (Cons 2 (Cons 3 Nil)))) ; [Cons 3 (Cons 2 (Cons 1 Nil))]
