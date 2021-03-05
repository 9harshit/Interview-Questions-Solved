"""mat = [[1,1,0],[2,1,1],[0,2,2]]
zero = {}
for i in range(3):
    for j in range(3):
        if mat[i][j] == 0:
            zero[i] = 1
            zero[j] = 1

for i in range(3):
    for j in range(3):
        if (i in zero) or (j in zero):
            mat[i][j] = 0

"""
"""# 3 1 Describe how you could use a single array to implement three stacks
stack = [-99] * 9
i = 0
n = 3
lenght = int(len(stack) / n)
pointer = []

for i in range(n):
    pointer.append(i * lenght)


def push(stackno, data):
    if pointer[stackno - 1] != (stackno * lenght):
        stack[pointer[stackno - 1]] = data
        pointer[stackno - 1] = pointer[stackno - 1] + 1
    else:
        print("Stack Full")


def pop(stackno):
    if pointer[stackno - 1] != ((stackno - 1) * lenght):
        print("Poped Element:", stack[pointer[stackno - 1] - 1])
        stack[pointer[stackno - 1] - 1] = -99
        pointer[stackno - 1] -= 1
    else:
        print("Stack Empty")


push(1, 1)
push(1, 2)
push(1, 3)
push(1, 4)
push(2, 2)
push(2, 2)
push(2, 2)
push(2, 2)"""

"""pop(1)
push(2,10)
push(3,20)
pop(2)
pop(3)
pop(1)
pop(2)
pop(1)
pop(3)"""


"""class queue:
    def __init__(self):
        self.s1 = []
        self.s2 = []
    def push(self, data):
        if (len(self.s1)==0):
            self.s1.append(data)
        else:
            while (len(self.s1)!=0):
                self.s2.append(self.s1.pop())
            self.s1.append(data)
            self.s1.append(self.s2)
    def peek():
        if (len(self.s1) != 0):
            print(self.s1.pop(0))
        
q = queue()
q.push(1)

q.peek() """

# anagram check

"""def anagram(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    print(str1, str2)
    
    str1 = str1.replace(" ", "")
    str2 = str2.replace(' ', '')
    
    if len(str1) != len(str2):
        return False

    char1 = {}
    char2 = {}
    for i in range(len(str1)):
        if str1[i] not in char1:
            char1[str1[i]] = 0
        else:
            char1[str1[i]]+=1
        if str2[i] not in char2:
            char2[str2[i]] = 0
        else:
            char2[str2[i]] += 1
            
    if char1 == char2:
        return True
    else:
        return False
        
from nose.tools import assert_equal

class AnagramTest(object):
    
    def test(self,sol):
        assert_equal(sol('go go go','gggooo'),True)
        assert_equal(sol('abc','cba'),True)
        assert_equal(sol('hi man','hi     man'),True)
        assert_equal(sol('aabbcc','aabbc'),False)
        assert_equal(sol('123','1 2'),False)
        print('ALL TEST CASES PASSED')

# Run Tests
t = AnagramTest()
t.test(anagram)"""


# Array Pair Sum

"""import math
def pair_sum(lst, k):

    no_pair = {}
    dup = 0

    for i in range(len(lst)-1):
        for j in range(i + 1, len(lst)):
            if lst[i] not in no_pair and lst[j] not in no_pair:
                if lst[i] + lst[j] == k:
                    no_pair[lst[i]] = lst[j]
                    no_pair[lst[j]] = lst[i]
                if lst[i] == lst[j]:
                    dup += 0.5
    print(no_pair)
    return math.ceil((len(no_pair)+ dup)/2)
from nose.tools import assert_equal

class TestPair(object):
    
    def test(self,sol):
        assert_equal(sol([1,9,2,8,3,7,4,6,5,5,13,14,11,13,-1],10),6)
        assert_equal(sol([1,2,3,1],3),1)
        assert_equal(sol([1,3,2,2],4),2)
        print('ALL TEST CASES PASSED')
        
#Run tests
t = TestPair()
t.test(pair_sum)"""

"""l = ['sat', 'bat', 'cat', 'mat'] 

# map() can listify the list of strings individually 
test = list(map(list, l)) 
print(test) 


"""

# Find the Missing Element


"""def finder(lst1, lst2):

    diff = sum(lst1) - sum(lst2)
    return diff


from nose.tools import assert_equal


class TestFinder(object):
    def test(self, sol):
        assert_equal(sol([5, 7, 7,0], [5, 7, 7]), 0)
        assert_equal(sol([1, 2, 3, 4, 5, 6, 7], [3, 7, 2, 1, 4, 6]), 5)
        assert_equal(sol([9, 8, 7, 6, 5, 4, 3, 2, 1], [9, 8, 7, 5, 4, 3, 2, 1]), 6)
        print("ALL TEST CASES PASSED")

One possible solution is computing the sum of all the numbers in arr1 and arr2, and subtracting arr2’s sum from array1’s sum. The difference is the missing number in arr2. However, this approach could be problematic if the arrays are too long, or the numbers are very large. Then overflow will occur while summing up the numbers.

By performing a very clever trick, we can achieve linear time and constant space complexity without any problems. Here it is: initialize a variable to 0, then XOR every element in the first and second arrays with that variable. In the end, the value of the variable is the result, missing element in array2.

def finder3(arr1, arr2): 
    result=arr[0]
    
    # Perform an XOR between the numbers in the arrays
    for num in arr1[1:]+arr2: 
        result^=num 
        print (result)
        
    return result 
# Run test
t = TestFinder()
t.test(finder)
"""
"""
IMPORTANT
# Largest Continuous Sum
def large_cont_sum(lst1):

    #negative array check
    if max(lst1) < 0:
        return max(lst1)

    max_sum = 0
    max_sub = 0

    if len(lst1) == 0:
        return 0

    
    for i in lst1:
        max_sub += i

        max_sum = max(max_sub, max_sum)

        max_sub = max(max_sub, 0)

    return max_sum


from nose.tools import assert_equal


class LargeContTest(object):
    def test(self, sol):
        assert_equal(sol([1, 2, -1, 3, 4, -1]), 9)
        assert_equal(sol([1, 2, -1, 3, 4, 10, 10, -10, -1]), 29)
        assert_equal(sol([-1, 1]), 1)
        print("ALL TEST CASES PASSED")


# Run Test
t = LargeContTest()
t.test(large_cont_sum)"""

# String Compression

"""def compress(s):
    if len(s) == 0:
        return ""
    
    dict1 = {}

    for i in range(len(s)):

        if s[i] not in dict1:
            dict1[s[i]] = 1

        else:
            dict1[s[i]] += 1
            
    new_string = ""

    for letter, count in dict1.items():
        new_string = new_string + str(letter) + str(count)
        
    print(new_string)
    return new_string

from nose.tools import assert_equal


class TestCompress(object):
    def test(self, sol):
        assert_equal(sol(""), "")
        assert_equal(sol("aaABBCC"), "a2A1B2C2")
        assert_equal(sol("AAABCCDDDDD"), "A3B1C2D5")
        print("ALL TEST CASES PASSED")


# Run Tests
t = TestCompress()
t.test(compress)"""

# Balanced Parentheses Check
# IMPORTANT from ipynb set variable
"""
def balance_check(s):

    if len(s)%2 != 0:
        return False
    paren = ["(", "{", "["]
    stack = []

    for i in s:
        if i in paren:
            stack.append(i)

        
        if i == ")" and len(stack) != 0:
            if "(" != stack.pop():
                return False

        if i == "]" and len(stack) != 0:
            if "[" != stack.pop():
                return False

        if i == "}" and len(stack) != 0:
            if "{" != stack.pop():
                return False
    
    if len(stack)==0 :
        return True
    else:
        return False

from nose.tools import assert_equal

class TestBalanceCheck(object):
    
    def test(self,sol):
        assert_equal(sol('[](){([[[]]])}('),False)
        assert_equal(sol('[{{{(())}}}]((()))'),True)
        assert_equal(sol('[[[]])]'),False)
        print ('ALL TEST CASES PASSED')
        
# Run Tests

t = TestBalanceCheck()
t.test(balance_check)"""


# Singly Linked List Cycle Check
# IMPORTANT
"""

class Node:
    def __init__(self, data):
        self.value = data
        self.nextnode = None


def cycle_check(a):
    if a.nextnode == None:
        return False

    ptr1 = a
    ptr2 = a

    while (ptr2.nextnode != None) and (ptr1.nextnode != None):
        ptr1 = ptr1.nextnode
        ptr2 = ptr2.nextnode.nextnode

        if ptr1.nextnode == ptr2.nextnode:

            return True

    return False


from nose.tools import assert_equal

# CREATE CYCLE LIST
a = Node(1)
b = Node(2)
c = Node(3)

a.nextnode = b
b.nextnode = c
c.nextnode = a  # Cycle Here!


# CREATE NON CYCLE LIST
x = Node(1)
y = Node(2)
z = Node(3)

x.nextnode = y
y.nextnode = z


#############
class TestCycleCheck(object):
    def test(self, sol):
        assert_equal(sol(a), True)
        assert_equal(sol(x), False)

        print("ALL TEST CASES PASSED")


# Run Tests

t = TestCycleCheck()
t.test(cycle_check)"""

# Linked List Reversal
"""class Node:
    def __init__(self, data):
        self.value = data
        self.nextnode = None


def reverse(a):
    if not a:
        return False

    else:
        next = None
        prev = None
        current = a
        while current:
            next = current.nextnode
            current.nextnode = prev
            prev = current
            current = next

    return prev


a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)

# Set up order a,b,c,d with values 1,2,3,4
a.nextnode = b
b.nextnode = c
c.nextnode = d

a = reverse(a)
ptr = a
while ptr:
    print(ptr.value)
    ptr = ptr.nextnode"""


# Linked List Nth to Last Node
"""class Node:
    def __init__(self, data):
        self.value = data
        self.nextnode = None


def nth_to_last_node(n, node):

    target_node = None
    if not node:
        return False

    else:
        i = 0
        ptr = node
        while i != n:
            current = ptr
            ptr = ptr.nextnode
            i += 1
        while current.nextnode:
            print(current.value, " ", node.value)
            current = current.nextnode
            node = node.nextnode
    target_node = node
    return target_node


from nose.tools import assert_equal

a = Node(1)
b = Node(2)
c = Node(3)
d = Node(4)
e = Node(5)

a.nextnode = b
b.nextnode = c
c.nextnode = d
d.nextnode = e

####


class TestNLast(object):
    def test(self, sol):

        assert_equal(sol(2, a), d)
        print("ALL TEST CASES PASSED")


# Run tests
t = TestNLast()
t.test(nth_to_last_node)"""


## RECUSRSION

# Write a recursive function which takes an integer and computes the cumulative sum of 0 to that integer


""" def rec_sum(n):

    if n == 0:
        return 0

    else:
        return n + rec_sum(n - 1)


print(rec_sum(4))

# Given an integer, create a function which returns the sum of all the individual digits in that integer. For example: if n = 4321, return 4+3+2+1


def sum_func(n):
    if n == 0:
        return 0

    else:
        return (n % 10) + sum_func(n // 10)


sum_func(4321)

#Create a function called word_split() which takes in a string phrase and a set list_of_words. The function will then determine if it is possible to split the string in a way in which words can be made from the list of words. You can assume the phrase will only contain words found in the dictionary if it is completely splittable.


def word_split(phrase, list_of_words, output=None):

    for w in list_of_words:

        
    
    return output
 """

# Reverse a String
"""def reverse(string):

    if string == "":
        return ""

    else:

        return string[len(string) - 1] + reverse(string[: len(string) - 1])


from nose.tools import assert_equal


class TestReverse(object):
    def test_rev(self, solution):
        assert_equal(solution("hello"), "olleh")
        assert_equal(solution("hello world"), "dlrow olleh")
        assert_equal(solution("123456789"), "987654321")

        print("PASSED ALL TEST CASES!")


# Run Tests
test = TestReverse()
test.test_rev(reverse)
"""

# STRING PERMUTATION
# IMPORTANT


""" def permute(string):

    if len(string) == 1:
        return string

    else:
        out = []
        for i in range(len(string)):
            for perm in permute(string[:i] + string[i + 1 :]):
                out.append(string[i] + perm)

    return out


from nose.tools import assert_equal


class TestPerm(object):
    def test(self, solution):

        assert_equal(
            sorted(solution("abc")), sorted(["abc", "acb", "bac", "bca", "cab", "cba"])
        )
        assert_equal(
            sorted(solution("dog")), sorted(["dog", "dgo", "odg", "ogd", "gdo", "god"])
        )

        print("All test cases passed.")


# Run Tests
t = TestPerm()
t.test(permute) """

# fibonaci
# IMPORTANT MEMOIZATION ENCAPSULATION
# sol 1

""" 
cache = {}


class Memo:
    def __init__(self, fun):
        self.fun = fun
        self.memory = {}

    def __call__(self, *args):
        if args not in self.memory:
            self.memory[args] = self.fun(*args)
            return self.memory[args]
        else:
            print("from memory")
            return self.memory[args]


def fib_rec(n):

    if n == 0:
        return 0

    if n == 1:
        return 1

    else:
        if n not in cache:
            cache[n] = fib_rec(n - 1) + fib_rec(n - 2)

        return cache[n]


# sol 2
def fib_rec2(n):
    if n == 0:
        return 0

    if n == 1:
        return 1
    a, b = 0, 1
    for i in range(n - 1):
        a, b = b, b + a

    return b


from nose.tools import assert_equal


class TestFib(object):
    def test(self, solution):
        assert_equal(solution(10), 55)
        assert_equal(solution(1), 1)
        assert_equal(solution(23), 28657)
        print("Passed all tests.")


t = TestFib()

t.test(fib_rec)
t.test(fib_rec2)

fib_rec = Memo(fib_rec)

t.test(fib_rec)

print(fib_rec(11))

"""
# COIN CHANGE PROBLEM SEE FOR RETURN STATMEMT
"""
class Memo:
    def __init__(self, f):
        self.f = f
        self.cache = {}

    def __call__(self, *args):
        k = str(args)
        if k not in self.cache:
            self.cache[k] = self.f(*args)
        return self.cache[k]


def rec_coin(n, coins):

    if n in coins:
        return 1

    if coins[-1] > n:
        return rec_coin(n, coins[:-1])

    else:
        return 1 + rec_coin(n - coins[-1], coins)


from nose.tools import assert_equal


class TestCoins(object):
    def check(self, solution):
        coins = [1, 5, 10, 25]
        coins.sort()
        assert_equal(solution(45, coins), 3)
        assert_equal(solution(23, coins), 5)
        assert_equal(solution(74, coins), 8)

        print("Passed all tests.")


# Run Test
test = TestCoins()
rec_coin = Memo(rec_coin)
test.check(rec_coin)
"""


# Imagine a robot sitting on the upper left hand corner of an NxN grid The robot can only move in two directions: right and down How many possible paths are there for the robot?
# FOLLOW UP
# Imagine certain squares are "off limits”, such that the robot can not step on them Design an algorithm to get all possible paths for the robot


""" def numberOfPaths(m, n):
    # If either given row number is first
    # or given column number is first
    if m == 1 or n == 1:
        return 1

    # If diagonal movements are allowed
    # then the last addition
    # is required.
    return numberOfPaths(m - 1, n) + numberOfPaths(m, n - 1)


# Driver program to test above function
m = 3
n = 3
print(numberOfPaths(m, n))
 """
# N QUEEN PROBLEM
""" n = 4


def check(board, row, col):

    for j in range(col):
        if board[row][j] == 1:
            return False

    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row, n), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True


def print_board(board):
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=" ")

        print("\n")


def place_queen(col, board):
    if col == n:
        return True

    for row in range(n):
        if check(board, row, col):
            board[row][col] = 1
            if place_queen(col + 1, board) == True:
                return True

            board[row][col] = 0

    return False


board = [[0 for i in range(n)] for i in range(n)]

place_queen(0, board)
print_board(board)
 """


### TREE ###

# Travesal

"""class BinaryTree:
    def __init__(self, obj):
        self.value = obj
        self.left = None
        self.right = None

    def insertleft(self, newobj):
        newobj = self.__class__(newobj)
        if self.left == None:
            self.left = newobj
        else:
            newobj.left = self.left
            self.left = newobj

    def insertright(self, newobj):
        newobj = self.__class__(newobj)
        if self.right == None:
            self.right = newobj
        else:
            newobj.right = self.left
            self.right = newobj


def inorder(tree):
    if tree:
        inorder(tree.left)
        print(tree.value)
        inorder(tree.right)


def preorder(tree):
    if tree:
        print(tree.value)
        preorder(tree.left)
        preorder(tree.right)


def postorder(tree):
    if tree:
        postorder(tree.left)
        postorder(tree.right)
        print(tree.value)


tree = BinaryTree(1)
tree.insertleft(2)
tree.insertright(3)
tree.left.insertleft(4)
tree.left.insertright(6)
tree.right.insertleft(5)
tree.right.insertright(7)
inorder(tree)
print(" ")
preorder(tree)
print(" ")
postorder(tree) 
"""


# 4 2 Given a directed graph, design an algorithm to find out whether there is a route be- tween two nodes

""" graph = {
    "A": set(["B", "C"]),
    "B": set(["A", "D", "E"]),
    "C": set(["A", "F"]),
    "D": set(["B"]),
    "E": set(["B", "F"]),
    "F": set(["C", "E"]),
}


def path_finder_bfs(graph, start, end):

    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        print(vertex, path)
        for next_vertex in graph[vertex] - set(path):
            if next_vertex == end:
                yield path + [next_vertex]
                # return True
            else:
                queue.append((next_vertex, path + [next_vertex]))


def path_finder_dfs(graph, start, end):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next_vertex in graph[vertex] - set(path):
            if next_vertex == end:
                yield path + [next_vertex]
                # return True
            else:
                stack.append((next_vertex, path + [next_vertex]))


print(list(path_finder_dfs(graph, "A", "F"))) """


# CHECK IF BINARY TREE IS BINARY SEARCH TREE

# IMPORTANT SECOND METHOD TO KEEP TRACK OF VALUES
""" class BinaryTree:
    def __init__(self, obj):
        self.value = obj
        self.left = None
        self.right = None

    def insertleft(self, newobj):
        newobj = self.__class__(newobj)
        if self.left == None:
            self.left = newobj
        else:
            newobj.left = self.left
            self.left = newobj

    def insertright(self, newobj):
        newobj = self.__class__(newobj)
        if self.right == None:
            self.right = newobj
        else:
            newobj.right = self.left
            self.right = newobj


# def checkBinaryTree(root):

#     if root:
#         if root.right and root.left:
#             if root.left.value < root.value and root.right.value >= root.value:
#                 checkBinaryTree(root.left)
#                 checkBinaryTree(root.right)
#             else:
#                 return False

#         if root.left and not root.right:
#             if root.left.value < root.value:
#                 checkBinaryTree(root.left)
#             else:
#                 return False

#         if not root.left and root.right:
#             if root.right.value >= root.value:
#                 checkBinaryTree(root.right)
#             else:
#                 return False

#     else:
#         raise KeyError("Tree is Empty")


def checkBinaryTree(tree, tree_values=None):
    if tree_values == None:
        tree_values = []
    if tree:
        checkBinaryTree(tree.left, tree_values)
        tree_values.append(tree.value)
        checkBinaryTree(tree.right, tree_values)

    return tree_values == sorted(tree_values)


tree = BinaryTree(10)
tree.insertleft(2)
tree.insertright(30)
tree.left.insertleft(1)
tree.left.insertright(6)
tree.right.insertleft(25)
tree.right.insertright(70)

if not checkBinaryTree(tree):
    print("Not a Binary Search Tree")

else:
    print("Binary Tree is Binary Search Tree")
 """

# Tree Level Order Print


""" class BinaryTree:
    def __init__(self, obj):
        self.value = obj
        self.left = None
        self.right = None

    def insertleft(self, newobj):
        newobj = self.__class__(newobj)
        if self.left == None:
            self.left = newobj
        else:
            newobj.left = self.left
            self.left = newobj

    def insertright(self, newobj):
        newobj = self.__class__(newobj)
        if self.right == None:
            self.right = newobj
        else:
            newobj.right = self.left
            self.right = newobj


nodes = []


def printlevelorder(tree):
    nodes.append(tree)
    currentlevel = 1
    nextlevel = 0
    while len(nodes) != 0:
        current = nodes.pop(0)
        currentlevel -= 1
        print(current.value, end=" ")

        if current.left:
            nodes.append(current.left)
            nextlevel += 1
        if current.right:
            nodes.append(current.right)
            nextlevel += 1
        if currentlevel == 0:
            print("")
            currentlevel, nextlevel = nextlevel, 0


tree = BinaryTree(1)
tree.insertleft(2)
tree.insertright(3)
tree.left.insertleft(4)
tree.right.insertleft(5)
tree.right.insertright(6)

list1 = printlevelorder(tree)
"""

# TRIM BST IMPORTANT JUPYTER


""" class BinaryTree:
    def __init__(self, obj, parent=None):
        self.value = obj
        self.left = None
        self.right = None
        self.parent = parent
        self.visit = 0

    def insertleft(self, newobj):
        newobj = self.__class__(newobj, self)
        if self.left == None:
            self.left = newobj
        else:
            newobj.left = self.left
            self.left = newobj

    def insertright(self, newobj):
        newobj = self.__class__(newobj, self)
        if self.right == None:
            self.right = newobj
        else:
            newobj.right = self.left
            self.right = newobj


nodes = []


def printlevelorder(tree):
    nodes.append(tree)
    currentlevel = 1
    nextlevel = 0
    while len(nodes) != 0:
        current = nodes.pop(0)
        currentlevel -= 1
        print(current.value, end=" ")

        if current.left:
            nodes.append(current.left)
            nextlevel += 1
        if current.right:
            nodes.append(current.right)
            nextlevel += 1
        if currentlevel == 0:
            print("")
            currentlevel, nextlevel = nextlevel, 0


def findsuccessor(tree):
    succ = None
    maxv = float("-inf")
    tree = tree.right
    while tree:
        if tree.value > maxv and (
            (not tree.left and not tree.right) or (not tree.left or not tree.right)
        ):
            succ = tree
        tree = tree.right

    return succ


def addchild(tree):
    if tree.visit == 0:
        flag = 0
        if tree.right:
            nodes.append(tree.right)
            flag = 1
        if tree.left:
            nodes.append(tree.left)
            flag = 1
        if flag == 1:
            tree.visit += 1


def trim(tree, minv, maxv):
    nodes.append(tree)
    while len(nodes) != 0:
        addchild(tree)
        check = nodes[-1]

        if check.visit >= 1:
            tree = nodes.pop(-1)
            print("visit", tree.value, " ", tree.visit)

        elif not check.left and not check.right:
            tree = nodes.pop(-1)
            print("leaf", tree.value, " ", tree.visit)
        else:
            tree = check
            continue

        if not tree.left and not tree.right:  # if leaf node check value
            if not (minv < tree.value <= maxv):
                if tree.parent.left == tree:
                    tree.parent.left = None
                else:
                    tree.parent.right = None
                if nodes:
                    tree = nodes[-1]
                continue

        if not tree.left or not tree.right:  # if node only has single child
            if not (minv < tree.value <= maxv):
                if tree.parent.left == tree:
                    if tree.left:
                        tree.parent.left = tree.left
                        tree.left.parent = tree.parent
                    else:
                        tree.parent.left = tree.right
                        tree.right.parent = tree.parent

                else:
                    if tree.left:
                        tree.parent.right = tree.left
                        tree.left.parent = tree.parent
                    else:
                        tree.parent.right = tree.right
                        tree.right.parent = tree.parent
                if nodes:
                    tree = nodes[-1]
                continue

        if tree.left and tree.right:
            if not (minv < tree.value <= maxv):
                succ = findsuccessor(tree)
                print("successor of ", tree.value, "is ", succ.value)
            if nodes:
                tree = nodes[-1]


tree = BinaryTree(8)
tree.insertleft(5)
tree.insertright(10)
tree.left.insertleft(3)
tree.left.insertright(6)
tree.left.right.insertleft(4)
tree.left.right.insertright(7)
tree.right.insertright(14)
tree.right.right.insertleft(13)
printlevelorder(tree)
print(" ")
trim(tree, , 13)
print(" ")
printlevelorder(tree) """

# Implement a function to check if a tree is balanced For the purposes of this question, a balanced tree is defined to be a tree such that no two leaf nodes differ in distance from the root by more than one


""" class BinaryTree:
    def __init__(self, obj, parent=None):
        self.value = obj
        self.left = None
        self.right = None

    def insertleft(self, newobj):
        newobj = self.__class__(newobj)
        if self.left == None:
            self.left = newobj
        else:
            newobj.left = self.left
            self.left = newobj

    def insertright(self, newobj):
        newobj = self.__class__(newobj)
        if self.right == None:
            self.right = newobj
        else:
            newobj.right = self.left
            self.right = newobj


def minDept(tree):
    if not tree:
        return 0
    else:
        return 1 + min(minDept(tree.left), minDept(tree.right))


def maxDept(tree):
    if not tree:
        return 0
    else:
        return 1 + max(maxDept(tree.left), maxDept(tree.right))


def checkbalance(tree):
    return abs(minDept(tree) - maxDept(tree)) <= 1


tree = BinaryTree(8)
tree.insertleft(5)
tree.insertright(10)
tree.left.insertleft(3)
tree.right.insertright(6)

checkbalance(tree)
 """

# Given a sorted (increasing order) array, write an algorithm to create a binary tree with minimal height


""" class BinaryTree:
    def __init__(self, obj, parent=None):
        self.value = obj
        self.left = None
        self.right = None


def insert(root, newobj):
    if newobj <= root.value:
        if root.left == None:
            root.left = BinaryTree(newobj)
        else:
            insert(root.left, newobj)
    if newobj > root.value:
        if root.right == None:
            root.right = BinaryTree(newobj)
        else:
            insert(root.right, newobj)


nodes = []


def printlevelorder(tree):
    nodes.append(tree)
    currentlevel = 1
    nextlevel = 0
    while len(nodes) != 0:
        current = nodes.pop(0)
        currentlevel -= 1
        print(current.value, end=" ")

        if current.left:
            nodes.append(current.left)
            nextlevel += 1
        if current.right:
            nodes.append(current.right)
            nextlevel += 1
        if currentlevel == 0:
            print("")
            currentlevel, nextlevel = nextlevel, 0


arr = [10, 50, 3, 2, 99]
arr.sort()


def createTree(arr):
    i = len(arr) // 2

    root = arr.pop(i)

    root = BinaryTree(root)

    for i in arr:
        insert(root, i)

    return root


printlevelorder(createTree(arr)) """

# Given a binary search tree, design an algorithm which creates a linked list of all the nodes at each depth (eg, if you have a tree with depth D, you’ll have D linked lists)

""" 
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class BinaryTree:
    def __init__(self, obj, parent=None):
        self.value = obj
        self.left = None
        self.right = None

    def insert(root, newobj):
        if newobj <= root.value:
            if root.left == None:
                root.left = BinaryTree(newobj)
            else:
                insert(root.left, newobj)
        if newobj > root.value:
            if root.right == None:
                root.right = BinaryTree(newobj)
            else:
                insert(root.right, newobj)


tree = BinaryTree(8)
tree.insert(5)
tree.insert(10)
tree.left.insert(3)
tree.right.insert(11)

nodes = []
nodes.append(tree)
currentlevet = 1
nextlevel = 0
answer = []
while len(nodes) != 0:
    currenttree = nodes.pop(0)
    if currenttree.left:
        nodes.append(currenttree.left)
        nextlevel += 1
    if currenttree.right:
        nodes.append(currenttree.right)
        nextlevel += 1
    if ll:
        ll.next = Node(currenttree.value)
        currentlevet -= 1
    else:
        ll = Node(currenttree.value)
        currentlevet -= 1
    if currentlevet == 0:
        currentlevet, nextlevel = nextlevel, 0
        answer.append(ll)
        ll = None

for i in answer:
    while i:
        print(i.value, end = " ")
        i = i.next

    print("") """

""" 
def selection_sort(arr):

    for i in range(len(arr) - 2):

        current = i
        for j in range(0, len(arr) - 1 - i):
            if arr[current] < arr[j]:
                current = j

        arr[current], arr[len(arr) - i - 1] = arr[len(arr) - i - 1], arr[current]

    print(arr)


selection_sort([2, 4, 6, 100, 9]) """


# 9 1 You are given two sorted arrays, A and B, and A has a large enough buffer at the end to hold B Write a method to merge B into A in sorted order
""" 

def merge_sort(a, b):
    i = len(a) - len(b) - 1
    j = len(b) - 1
    k = len(a) - 1

    while (k >= 0) and j >= 0 and i >= 0:
        if a[i] > b[j]:
            a[k] = a[i]
            i -= 1
        else:
            a[k] = b[j]
            j -= 1
        k -= 1

    while j >= 0:
        a[k] = b[j]
        j -= 1
        k -= 1
    print(a)


merge_sort([3, 5, 6, None, None], [1, 10]) """

# 9 2 Write a method to sort an array of strings so that all the anagrams are next to each other
""" 

def sort_anagram(arr):
    dict_words = {}
    grp = []
    answer = []
    for i in arr:
        k = tuple(set(i))
        if k not in dict_words:
            dict_words[k] = i
        else:
            dict_words[k] = dict_words[k] + " " + i

    for i in dict_words.values():
        grp.append(sorted(i.split(" ")))

    print(grp)
    for i in grp:
        answer = answer + sorted(i)

    print(sorted(answer))


sort_anagram(["god", "abc", "dog", "cab", "man"]) """

# 9 3 Given a sorted array of n integers that has been rotated an unknown number of times,giveanO(logn)algorithmthatfindsanelementinthearray Youmayassume that the array was originally sorted in increasing order
# EXAMPLE: Input: find 5 in array (15 16 19 20 25 1 3 4 5 7 10 14) Output: 8 (the index of 5 in the array)

""" 
def binary_search(arr, ele):

    first = 0
    last = len(arr) - 1
    mid = 0 + len(arr) // 2
 w
    if not arr:
        return False
    while mid >= 0 and first <= last:
        print(first, " ", arr[mid], " ", last)
        print(arr[first : last + 1])

        if arr[mid] == ele:
            return mid

        if arr[first] < arr[mid] < arr[last]:
            if ele > arr[mid]:
                first = mid + 1
            else:
                last = mid - 1

        else:
            if arr[mid] < arr[last] or ele > arr[mid]:
                first = mid + 1
            else:
                last = mid - 1

        mid = (last + first) // 2

    return False


print(binary_search([15, 16, 19, 20, 25, 1, 3, 4, 5, 7, 10, 14], 1)) """


# 9 5 Given a sorted array of strings which is interspersed with empty strings, write a meth- od to find the location of a given string
# Example: find "ball” in ["at”, "”, "”, "”, "ball”, "”, "”, "car”, "”, "”, "dad”, "”, "”] will return 4 Example: find "ballcar” in ["at”, "”, "”, "”, "”, "ball”, "car”, "”, "”, "dad”, "”, "”] will return -1


""" def string_binary(arr, ele):

    i = len(arr) // 2
    print(arr, " ", i)
    if arr[i] == ele:
        return i

    elif i <= 0:
        return -1

    else:
        j = i + 1
        while j <= len(arr) - 1:
            if arr[j] == ele:
                return j
            j += 1

        return string_binary(arr[:i], ele)


print(
    string_binary(
        ["at", "", "", "", "ball", "", "", "car", "", "", "dad", "", ""], "ball"
    )
)

print(
    string_binary(
        ["at", "", "", "", "", "ball", "car", "", "", "dad", "", ""], "ballcard"
    )
)
 """
# sort
""" 
class person:
    def __init__(self, ht, wth):
        self.ht = ht
        self.wth = wth


def sort_circus(arr):
    arr = sorted(arr, key=lambda person: person.ht)
    for i in range(len(arr) - 1):
        if arr[i].ht == arr[i + 1].ht:
            if arr[i].wth > arr[i + 1].wth:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]

    answer = []
    for i in range(len(arr) - 1):
        if arr[i].ht > arr[i + 1].ht or arr[i].wth > arr[i + 1].wth:
            print("Not fit", arr[i].ht, " ", arr[i].wth)
        else:
            answer.append(arr[i])

    answer.append(arr[-1])

    for i in answer:
        print(i.ht, " ", i.wth)

    return len(answer)


arr = [
    person(65, 100),
    person(70, 150),
    person(56, 90),
    person(75, 190),
    person(60, 95),
    person(68, 110),
]

sort_circus(arr) """

""" 
def rem(str1):
    stack = []
    dict1 = {}

    for i in str1:
        if i == " ":
            stack.append(i)
            continue
        if i not in dict1:
            dict1[i] = 1
            stack.append(i)

    return "".join(stack)


rem("tree travesal")


def sum1(lst, target):
    dict1 = {
    }
    for i in lst:
        if i not in dict1:
            answer = target - i
            dict1[answer] = 1
        else:
            return True
    return False

sum1([5,3,1,9], 12)
        

def uni(lst):

    for i in lst:
        if len(str(i)) == len(set(str(i))):
            return i
    
    return None

uni([111,22,3])


def solution(unsorted_prices,max_price):
    
    # list of 0s at indices 0 to max_price
    prices_to_counts = [0]* (max_price+1)
    
    # populate prices
    for price in unsorted_prices:
        prices_to_counts[price] +=1
        
    # populate final sorted prices
    sorted_prices = []
    
    # For each price in prices_to_counts
    for price,count in enumerate(prices_to_counts):
        
        # for the number of times the element occurs
        for time in range(count):
            
            # add it to the sorted price list
            sorted_prices.append(price)
            
    return sorted_prices


solution([1,1,3,7,9,5,31,3], 31)


 """


