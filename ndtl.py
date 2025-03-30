from abc import abstractmethod
from collections import deque
import numpy as np
from scipy.stats import norm

class FormulaBase:
    def __init__(self, id):
        self.id = id

    @abstractmethod
    def evaluate(self, graph, node):
        """evaluate the formula rooted at self"""

    @abstractmethod
    def toString(self):
        """print the formula rooted at self"""

class NDTLFormula(FormulaBase):

    def __init__(self, id):
        super().__init__(id)

class NDTLUnaryOperator(NDTLFormula):

    def __init__(self, id, child):
        super().__init__(id)
        self.child = child

class NDTLBinaryOperator(NDTLFormula):

    def __init__(self, id, left, right):
        super().__init__(id)
        self.left = left
        self.right = right


class NDTLAtomic(NDTLFormula):

    def __init__(self, id):
        super().__init__(id)

class NDTLNode(NDTLAtomic):

    def __init__(self, id, predicate):
        super().__init__(id)
        self.predicate = predicate

    def evaluate(self, graph, node):
        return graph.nodes[node]["type"] == self.predicate

    def toString(self):
        return(self.predicate)

class NDTLAggregateFormula(NDTLAtomic):
    def __init__(self, id, aggregate1, aggregate2, comparator, threshold, mean=None, std=None):
        super().__init__(id)
        self.aggregate1 = aggregate1
        self.aggregate2 = aggregate2
        self.comparator = comparator
        self.threshold = threshold
        self.mean = mean
        self.std = std


    def evaluate(self, graph, node):
        result = self.aggregate1.evaluate(graph, node)
        if result == "nan":
            return result
        if not (self.aggregate2 is None):
            result2 = self.aggregate2.evaluate(graph, node)
            if result2 == "nan":
                return result2
            result += result2
        #print(self.toString(), str(result)+self.comparator+str(self.threshold))
        if self.threshold is not None:
            if self.comparator == "<=":
                return result <= self.threshold
            elif self.comparator == ">=":
                return result >= self.threshold
            elif self.comparator == "<":
                return result < self.threshold
            elif self.comparator == ">":
                return result > self.threshold
            else:
                raise ValueError(f"Unknown operator: {self.comparator}")
        elif self.mean is not None:
            #z = abs((np.log(result+1) - self.mean) / self.std)
            z = abs((result  - self.mean) / self.std)
            return z <= 3

    def toString(self):
        if not (self.aggregate2 is None):
            return("("+self.aggregate1.toString()+self.aggregate2.toString()+self.comparator+str(self.threshold)+")")
        else:
            return("("+self.aggregate1.toString()+self.comparator+str(self.threshold)+")")


class NDTLAggregate(NDTLAtomic):
    def __init__(self, id, aggregate, attribute, neighbors, neighborhood):
        super().__init__(id)
        self.aggregate = aggregate
        self.attribute = attribute
        self.neighbors = neighbors
        self.neighborhood = neighborhood

    def toString(self):
        #print (type(self.neighbors), self.neighbors)
        return(self.aggregate+self.attribute+"(Gamma("+self.neighbors.toString()+","+str(self.neighborhood)+"))")

    def evaluate(self, graph, node):
        neighborhoodNodes = self.get_neighborhood(graph, node, self.neighborhood)
        #print(self.toString(), node, str(len(neighborhoodNodes)), neighborhoodNodes, [(graph.nodes[n],graph.nodes[n]["type"]) for n in neighborhoodNodes])

        # Filter nodes by the neighbor type
        filtered_nodes = [n for n in neighborhoodNodes if graph.nodes[n]["type"] == self.neighbors.toString()]
        #print ([(n,graph.nodes[n]["type"], self.neighbors, graph.nodes[n]["type"] == self.neighbors) for n in neighborhoodNodes])
        #print(self.toString(), node, str(len(filtered_nodes)))

        # Extract the attribute values (e.g., delta or sigma) for the filtered nodes
        #print([graph.nodes[n] for n in filtered_nodes])
        values = [graph.nodes[n][self.attribute] for n in filtered_nodes]
        #print(self.toString(), node, values)

        if len(values)==0:
            return "nan"

        # Perform the aggregate operation
        if self.aggregate == "avg":
            result = sum(values) / len(values) if values else float("inf")
        elif self.aggregate == "max":
            result = max(values) if values else float("-inf")
        elif self.aggregate == "min":
            result = min(values) if values else float("inf")
        else:
            raise ValueError(f"Unknown aggregate function: {self.aggregate}")

        return result

    def get_neighborhood(self, graph, node, neighborhood_size):

        if neighborhood_size == "inf":
            # Perform BFS to find all descendants
            visited = set()
            visited.add(node)
            queue = deque(graph.successors(node))
            descendants = []

            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    descendants.append(current)
                    queue.extend(graph.successors(current))

            return descendants

        elif isinstance(neighborhood_size, int) and neighborhood_size >= 1:
            # Perform BFS up to the given depth
            visited = set()
            queue = deque([(node, 0)])  # (current_node, current_depth)
            neighborhood = []

            while queue:
                current, depth = queue.popleft()
                if current not in visited and depth <= neighborhood_size:
                    visited.add(current)
                    if depth>0:
                        neighborhood.append(current)
                    if depth < neighborhood_size:
                        queue.extend((neighbor, depth + 1) for neighbor in graph.successors(current))

            return neighborhood

        else:
            raise ValueError(f"Invalid neighborhood size: {neighborhood_size}")

class NDTLArithmeticAggregate(NDTLAtomic):
    def __init__(self, id, operator, aggregate):
        super().__init__(id)
        self.operator = operator
        self.aggregate = aggregate

    def toString(self):
        return(self.operator+self.aggregate.toString())

    def evaluate(self, graph, node):
        result = self.aggregate.evaluate(graph, node)
        if result == "nan":
            return result
        if self.operator == "-":
            return -result
        elif self.operator == "+":
            return result
        else:
            raise ValueError(f"Unknown aggregate function: {self.operator}")

class NDTLNot(NDTLUnaryOperator):

    def __init__(self, id, child):
        super().__init__(id, child)

    def toString(self):
        return("NOT ("+self.child.toString()+")")

    def evaluate(self, graph, node):
        result = self.child.evaluate(graph, node)
        if result == "nan":
            return result
        return not result

class NDTLAnd(NDTLBinaryOperator):

    def __init__(self, id, left, right):
        super().__init__(id, left, right)

    def toString(self):
        return("("+self.left.toString()+" AND "+self.right.toString()+")")

    def evaluate(self, graph, node):
        resultl = self.left.evaluate(graph, node)
        resultr = self.right.evaluate(graph, node)
        if resultl == "nan" or resultr == "nan":
            return "nan"
        return resultl and resultr

class NDTLOr(NDTLBinaryOperator):

    def __init__(self, id, left, right):
        super().__init__(id, left, right)

    def toString(self):
        return("("+self.left.toString()+" OR "+self.right.toString()+")")


    def evaluate(self, graph, node):
        resultl = self.left.evaluate(graph, node)
        resultr = self.right.evaluate(graph, node)
        if resultl == "nan" or resultr == "nan":
            return "nan"
        return resultl or resultr


class NDTLImplies(NDTLBinaryOperator):

    def __init__(self, id, left, right):
        super().__init__(id, left, right)

    def toString(self):
        return("("+self.left.toString()+" -> "+self.right.toString()+")")


    def evaluate(self, graph, node):
        #print(f"In evaluate {self.toString()} for node {node}")
        resultl = self.left.evaluate(graph, node)
        resultr = self.right.evaluate(graph, node)
        if resultl == "nan" or resultr == "nan":
            return "nan"
        return (not resultl) or resultr
    

class NDTLEquivalence(NDTLBinaryOperator):

    def __init__(self, id, left, right):
        super().__init__(id, left, right)

    def toString(self):
        return("("+self.left.toString()+" <-> "+self.right.toString()+")")


    def evaluate(self, graph, node):
        resultl = self.left.evaluate(graph, node)
        resultr = self.right.evaluate(graph, node)
        if resultl == "nan" or resultr == "nan":
            return "nan"
        return  resultl == resultr
    
class NDTLNext(NDTLUnaryOperator):

    def __init__(self, id, child, th=1):
        super().__init__(id, child)
        self.th = th

    def toString(self):
        return("X("+self.child.toString()+")")

    def evaluate(self, graph, node):
        neighbors = list(graph.successors(node))
        return all(
            self.child.evaluate(graph, n)
            for n in neighbors
            if self.child.evaluate(graph, n) != "nan"
        )


class NDTLEventually(NDTLUnaryOperator):

    def __init__(self, id, child, th=1):
        super().__init__(id, child)
        self.th = th

    def toString(self):
        return("F("+self.child.toString()+")")

    def evaluate(self, graph, node):
        successors = list(graph.successors(node))

        # Perform a DFS with path tracking for each successor
        for successor in successors:
            stack = [(successor, [])]  # Stack holds (current_node, path_so_far)
            while stack:
                current, path = stack.pop()
                new_path = path + [current]

                # If the current node is a leaf, check if the path satisfies the inner formula
                if len(list(graph.successors(current))) == 0:  # Leaf node
                    if all(self.child.evaluate(graph, n) for n in new_path  if self.child.evaluate(graph, n) != "nan"):
                        return True

                # Otherwise, continue exploring the subgraph
                stack.extend((child, new_path) for child in graph.successors(current))

        return False
    
class NDTLAlways(NDTLUnaryOperator):

    def __init__(self, id, child, th=1):
        super().__init__(id, child)
        self.th=th

    def toString(self):
        #print(self.child)
        #print("G(", self.child.string(), ")")
        #child_str = self.child.toString()
        return ("G("+self.child.toString()+")")

    #def evaluate(self, graph, node):
    #    print(f"In evaluate {self.toString()} for node {node}")
    #    successors = list(graph.successors(node))
    #    print(f"In evaluate {self.toString()} for node {node} successors size {len(successors)}")

    #    i=0
        # Perform a DFS for each successor to ensure all nodes satisfy the formula
    #    for successor in successors:
    #        stack = [successor]
            #if(i):
    #        i+=1
    #        print(f"In evaluate {self.toString()} for node {node} successors  {(i)}")
    #        while len(stack)>0:
    #            current = stack.pop(0)
    #            print(f"---> {current}")
                #if (i):
                #    print(f"In evaluate {self.toString()} for node {node} successors  {(stack)}")
                #    print(f"In evaluate {self.toString()} for node {node} successors size {len(stack)}")
    #            if self.child.evaluate(graph,current)!="nan" and not self.child.evaluate(graph,current):
    #                return False
    #            print(f"{graph} graph {current} succ: {list(graph.successors(current))}")
    #            stack.extend(graph.successors(current)) # Add all children of the current node
                #if (i):
                #    print(f"In evaluate {self.toString()} for node {node} successors size {len(stack)}")
                #    i = False
                #print(f"In evaluate {self.toString()} for node {node} successors size {len(successors)}")
    #    return True

    def evaluate(self, graph, node):
        #print(f"In evaluate {self.toString()} for node {node}")
        stack = [node]
        total_nodes = 0
        valid_nodes = 0
        #print(f"In evaluate {self.toString()} for node {node} successors size {len(successors)}")

        while stack:
            current = stack.pop(0)
            result = self.child.evaluate(graph, current)

            if result != "nan":
                total_nodes += 1
                if result:
                    valid_nodes += 1


            #print(f"---> {current} {graph.nodes[current]}")
                #if (i):
                #    print(f"In evaluate {self.toString()} for node {node} successors  {(stack)}")
                #    print(f"In evaluate {self.toString()} for node {node} successors size {len(stack)}")
            #print(f"{graph} graph {current} succ: {list(graph.successors(current))}")
            stack.extend(graph.successors(current)) # Add all children of the current node
                #if (i):
                #    print(f"In evaluate {self.toString()} for node {node} successors size {len(stack)}")
                #    i = False
                #print(f"In evaluate {self.toString()} for node {node} successors size {len(successors)}")
        if total_nodes != 0 and (valid_nodes / total_nodes) < self.th:
            return False
        return True

