
from pathlib import Path
from lark import Transformer, Lark
from ndtl import *



class NDTLTransformer(Transformer):
   
    def start(self, args):  # type: ignore
        #print ("START",args)
        return args[0]

    def ndtl_formula(self, args):
        #print ("ndtl_formula",args)
        return args[0]

    def ndtl_unaryop(self, args):
        #print ("ndtl_unaryop",args)
        return args[0]
    
    def aggregate_formula(self, args):
        #print ("aggregate_formula",args)
        if len(args) ==3 :
            _,comp,th=args
            return NDTLAggregateFormula(0,args[0],None,str(comp), float(th))
        elif len(args) == 4 and str(args[1])!="=":
            _,_,comp,th = args
            return NDTLAggregateFormula(0, args[0],args[1], str(comp), float(th))
        elif len(args) == 4:
            _, comp, mean, std = args
            return NDTLAggregateFormula(0, args[0], None, str(comp),None,float(mean), float(std))
        elif len(args) == 5:
            _, _, comp, mean, std = args
            return NDTLAggregateFormula(0, args[0],args[1],str(comp), None, float(mean), float(std))

    def arithmetic_expression(self, args):
        #print("arithmetic_expression", args)
        op, _ = args
        return NDTLArithmeticAggregate(0, str(op), args[1])
    
    def ndtl_atom(self, args):
        #print ("ndtl_atom",args)
        return args[0]
    
    def ndtl_equivalence(self, args):
        #print ("ndtl_equivalence",args)
        return NDTLEquivalence(0, args[0], args[2])
        #return {"type": "eq", "left": args[0], "right": args[2]}
        
        
    def ndtl_implication(self, args):
        #print ("ndtl_implication",args)
        return NDTLImplies(0, args[0], args[2])
        #return {"type": "imp", "left": args[0], "right": args[2]}

    def ndtl_or(self, args):
        #print ("ndtl_or",args)
        return NDTLOr(0, args[0], args[2])
        #return {"type": "or", "left": args[0], "right": args[2]}

    def ndtl_and(self, args):
        #print ("ndtl_and",args)
        return NDTLAnd(0, args[0], args[2])
        #return {"type": "and", "left": args[0], "right": args[2]}
        
    def ndtl_always(self, args):
        #print ("ndtl_always",args)
        if (len(args) == 3):
            return NDTLAlways(0, args[2], float(args[1]))
        else:
            return NDTLAlways(0, args[1])
        #return {"type": "G", "inner": args[1]}

    def ndtl_eventually(self, args):
        if (len(args) == 3):
            return NDTLEventually(0, args[2], float(args[1]))
        else:
            return NDTLEventually(0, args[1])
        #return {"type": "F", "inner": args[1]}

    def ndtl_next(self, args):
        if (len(args) == 3):
            return NDTLNext(0, args[2], float(args[1]))
        else:
            return NDTLNext(0, args[1])
        #return {"type": "X", "inner": args[1]}
    
    def ndtl_not(self, args):
        return NDTLNot(0, args[1])
        #return {"type": "not", "inner": args[1]}


    def ndtl_wrapped(self, args):
        #print("wrapped", args)
        _, formula, _ = args
        return formula
    
    def aggregate(self, args):
        #print ("aggregate",args)
        typ, attribute, node, value = args
        return NDTLAggregate(0, str(typ), str(attribute), node, int(value) if value != "inf" else "inf")
        #return {"type": "aggregate","function": str(typ),"attribute": str(attribute),"neighbors": str(node),"neighborhood": str(value) if value != "inf" else "inf"}

    def node(self, args):
        #print ("node",args)
        return NDTLNode(0,str(args[0]))
        #return {"type": "node", "value": str(args[0])}


    

if __name__ == "__main__":
    ndtl_parser = Lark(
        Path("ndtl_grammar.lark").open("r").read(), parser="lalr", start="start"
    )
    test_input = "G (post -> (maxsigma(Gamma(tweet, 1)) - minsigma(Gamma(tweet, 1)) <= 5))"

    # Parse the input
    try:
        tree = ndtl_parser.parse(test_input)
        print("Parse Tree:")
        print(tree.pretty())
        
        transformer = NDTLTransformer()
        transformed_tree = transformer.transform(tree)
        print("Transformed Tree:")
        print(transformed_tree.string())

        
    except Exception as e:
        print(f"Error parsing input: {e}")
