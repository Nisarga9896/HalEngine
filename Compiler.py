from llvmlite import ir
import os
from AST import Node,NodeType,Program,Statement,Expression, VarStatement,IdentifierLiteral,ReturnStatement,AssignStatement,CallExpression,InputExpression,NullLiteral
from AST import ExpressionStatement,InfixExpression,IntegerLiteral,FloatLiteral, BlockStatement,FunctionStatement,IfStatement,BooleanLiteral,ArrayLiteral,RefExpression,DerefExpression
from AST import FunctionParameter,StringLiteral,WhileStatement,BreakStatement,ContinueStatement,PrefixExpression,PostfixExpression,LoadStatement,ArrayAccessExpression, StructInstanceExpression,StructAccessExpression,StructStatement
from typing import List, cast
from Environment import Environment
from typing import Optional
from lexer import Lexer
from Parser import Parser

# Built-in headers (minimal set)
BUILTIN_HEADERS = {
    
}


class Compiler:
    def __init__(self)-> None:
        self.errors: list[str] = []
        self.array_lengths: dict[str, int] = {}
        self.struct_types: dict[str, ir.IdentifiedStructType] = {}
        self.struct_layouts: dict[str, dict[str, int]] = {}
        self.type_map:dict[str,ir.Type]={
            'int':ir.IntType(32),
            'float':ir.FloatType(),
            'void':ir.VoidType(),
            'bool':ir.IntType(1),
            'str':ir.PointerType(ir.IntType(8)),
            'null': ir.VoidType()
        }
        self.counter:int=0
        self.module:ir.Module=ir.Module('main')
        self.true_str = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 5), name="true_str")
        self.true_str.global_constant = True
        self.true_str.linkage = 'internal'
        self.true_str.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 5), bytearray(b"true\0")) # type: ignore

        self.false_str = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 6), name="false_str")
        self.false_str.global_constant = True
        self.false_str.linkage = 'internal'
        self.false_str.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 6), bytearray(b"false\0")) # type: ignore
        self.builder:ir.IRBuilder=ir.IRBuilder()
        self.env:Environment=Environment()
        self.__initialize_builtins()
        self.breakpoints:list[ir.Block]=[]
        self.continues:list[ir.Block]=[]
        self.global_parsed_pallets:dict[str,Program]={}

    def __initialize_builtins(self)->None:
        def __init_print()->ir.Function:
            fnty:ir.FunctionType=ir.FunctionType(
                self.type_map['int'],
                [ir.IntType(8).as_pointer()],
                var_arg=True
            )
            return ir.Function(self.module,fnty,'printf')
        def __init_scanf()->ir.Function:
            fnty:ir.FunctionType=ir.FunctionType(
                self.type_map['int'],
                [ir.IntType(8).as_pointer()],
                var_arg=True
            )
            return ir.Function(self.module,fnty,'scanf')


        def __init_booleans()->tuple[ir.GlobalVariable,ir.GlobalVariable]:
            bool_type:ir.Type=self.type_map['bool']
            true_var=ir.GlobalVariable(self.module,bool_type,'true')
            true_var.initializer=ir.Constant(bool_type,1) # type: ignore
            true_var.global_constant=True
            false_var=ir.GlobalVariable(self.module,bool_type,'false')
            false_var.initializer=ir.Constant(bool_type,0) # type: ignore
            false_var.global_constant=True

            return true_var,false_var
        
        def __init_null()->ir.GlobalVariable:
            null_type = self.type_map['int'].as_pointer() # Using a null pointer for 'null'
            null_var = ir.GlobalVariable(self.module, null_type, 'null')
            null_var.initializer = ir.Constant(null_type, None) # type: ignore
            null_var.global_constant = True
            return null_var
        
                # len() for arrays

        def __init_malloc()->ir.Function:
            fnty:ir.FunctionType=ir.FunctionType(
                ir.IntType(8).as_pointer(),
                [ir.IntType(32)],
                var_arg=False
            )
            return ir.Function(self.module,fnty,'malloc')
        
        def __builtin_len(params: list[ir.Value], return_type: ir.Type):
            if len(params) != 1:
                self.report_error("len() requires exactly 1 parameter.")
                return None

            array_val = params[0]
            if array_val is None: 
                self.errors.append("ddd")
                return 
            array_type = getattr(array_val, "type", None)


            # Case: LLVM ArrayType
            if isinstance(array_type, ir.ArrayType):
                return ir.Constant(ir.IntType(32), array_type.count)

            # Case: pointer to ArrayType ([N x T]*)
            if isinstance(array_type, ir.PointerType) and isinstance(array_type.pointee, ir.ArrayType):  # type: ignore
                return ir.Constant(ir.IntType(32), array_type.pointee.count)  # type: ignore

            self.report_error("len() only works on fixed-size arrays right now.")
            return None

        self.builtin_functions = {}
        self.builtin_functions["len"] = __builtin_len


        
        #print
        self.env.define('print',__init_print(),ir.IntType(32))
        self.env.define('input',__init_scanf(),ir.IntType(32))
        self.env.define('reserve', __init_malloc(), ir.IntType(8).as_pointer())

        true_var,false_var=__init_booleans()
        self.env.define('true',true_var,true_var.type)
        self.env.define('false',false_var,false_var.type)

        null_var = __init_null()
        self.env.define('null', null_var, null_var.type)
        
    def increment_counter(self)->int:
        self.counter+=1
        return self.counter

    def compile(self, node: Node) -> None:
        match node.type():
            case NodeType.Program:
                self.visit_program(cast(Program, node))

            case NodeType.ExpressionStatement:
                self.visit_expression_statement(cast(ExpressionStatement,node))

            case NodeType.VarStatement:
                self.visit_var_statement(cast(VarStatement,node))
            case NodeType.InfixExpression:
                self.visit_infix_expression(cast(InfixExpression, node))
            case NodeType.PostfixExpression:
                self.visit_postfix_expression(cast(PostfixExpression, node))
            case NodeType.FunctionStatement:
                self.visit_function_statement(cast(FunctionStatement,node))
            case NodeType.BlockStatement:
                self.visit_block_statement(cast(BlockStatement,node))
            case NodeType.ReturnStatement:
                self.visit_return_statement(cast(ReturnStatement,node))
            case NodeType.AssignStatement:
                self.visit_assign_statement(cast(AssignStatement,node))
            case NodeType.IfStatement:
                self.visit_if_statement(cast(IfStatement,node))
            case NodeType.WhileStatement:
                self.visit_while_statement(cast(WhileStatement,node))
            
            case NodeType.StructStatement:
                self.visit_struct_definition_statement(cast(StructStatement, node))
            
            case NodeType.CallExpression:
                self.visit_call_expression(cast(CallExpression,node))
            case NodeType.BreakStatement:
                self.visit_break_statement(cast(BreakStatement,node))
            case NodeType.ConinueStatement:
                self.visit_continue_statement(cast(ContinueStatement,node))
            case NodeType.LoadStatement:
                self.visit_load_statement(cast(LoadStatement,node))
            case NodeType.NullLiteral: 
                self.visit_null_literal(cast(NullLiteral, node))
            
    # In Compiler.py, add these new methods to the class

    # In Compiler.py

    def visit_struct_definition_statement(self, node: StructStatement) -> None:
        struct_name = node.name.value
        if struct_name is None:
            self.report_error("Struct definition is missing a name.")
            return
        if struct_name in self.struct_types:
            self.report_error(f"Struct '{struct_name}' is already defined.")
            return

        # Create an opaque struct type first. This is a placeholder.
        struct_type = self.module.context.get_identified_type(struct_name)
        
        # ADDED: This is the crucial missing step.
        # You must register the struct type so you can find it later during instantiation.
        self.struct_types[struct_name] = struct_type
        
        self.struct_layouts[struct_name] = {}

        member_types = []
        for i, member_var_stmt in enumerate(node.members):
            member_name = cast(IdentifierLiteral, member_var_stmt.name).value
            if member_name is None:
                self.report_error(f"Invalid member in struct '{struct_name}'.")
                continue

            # Your logic for determining member types is fine.
            if member_var_stmt.value_type is None:
                # Defaulting 'var' members to 'int'
                member_type = self.type_map['int']
            else:
                # This would handle types like 'array', etc., if you add them later
                member_type = self.type_map.get(member_var_stmt.value_type, self.type_map['int'])

            member_types.append(member_type)
            self.struct_layouts[struct_name][member_name] = i
        
        # Now, set the body of the struct with the resolved member types.
        struct_type.set_body(*member_types)


    def visit_struct_instantiation(self, node: StructInstanceExpression) -> Optional[tuple[ir.Value, ir.Type]]:
        """
        Handles struct instantiation, e.g., Point(). Allocates memory and returns a pointer.
        """
        struct_name = node.struct_name.value
        if struct_name is None:
            self.report_error("Struct instantiation is missing a name.")
            return None

        struct_type = self.struct_types.get(struct_name)
        if struct_type is None:
            self.report_error(f"Attempting to instantiate unknown struct type '{struct_name}'.")
            return None

        # Allocate memory for the struct on the stack.
        ptr = self.builder.alloca(struct_type, name=f"{struct_name}_instance")
        
        # The "value" of a struct instantiation is the pointer to its memory.
        return ptr, ir.PointerType(struct_type)

    def visit_member_access(self, node: StructAccessExpression) -> Optional[tuple[ir.Value, ir.Type]]:
        # ... (This method is correct, no changes needed)
        member_name = node.member_name.value
        if member_name is None:
            self.report_error("Member access is missing a member name.")
            return None

        obj_resolved = self.resolve_value(node.struct_name)
        if obj_resolved is None:
            self.report_error("Could not resolve object in member access.")
            return None
        obj_ptr, obj_type = obj_resolved

        if not isinstance(obj_type, ir.PointerType) or not isinstance(obj_type.pointee, ir.IdentifiedStructType): # type: ignore
            self.report_error("Member access '.' operator can only be used on struct instances.")
            return None

        struct_type = cast(ir.IdentifiedStructType, obj_type.pointee) # type: ignore
        struct_name = struct_type.name
        layout = self.struct_layouts.get(struct_name)
        member_index = layout.get(member_name) if layout else None

        if member_index is None:
            self.report_error(f"Struct '{struct_name}' has no member named '{member_name}'.")
            return None

        zero = ir.Constant(ir.IntType(32), 0)
        member_ptr = self.builder.gep(obj_ptr, [zero, ir.Constant(ir.IntType(32), member_index)], inbounds=True, name=f"{member_name}_ptr")

        loaded_value = self.builder.load(member_ptr, name=member_name)
        return loaded_value, loaded_value.type 

        
    def visit_program(self,node:Program) ->None:
        for stmt in node.statements:
            self.compile(stmt)
        

    def visit_expression_statement(self, node: ExpressionStatement) -> None:
        if node.expr is not None:
            if isinstance(node.expr, StructStatement):
                self.visit_struct_definition_statement(node.expr)
            else:
                self.resolve_value(node.expr)
    
            
    def visit_var_statement(self, node: VarStatement) -> None:
        if node.name is None or not isinstance(node.name, IdentifierLiteral):
            raise ValueError("Variable name must be a non-null IdentifierLiteral")
        
        identifier = cast(IdentifierLiteral, node.name)
        if identifier.value is None:
            raise ValueError("Identifier name cannot be None")
        name: str = identifier.value  
        if node.value is None:
            raise ValueError(f"Variable '{name}' has no assigned value.")
    
        value_expr: Expression = node.value
        value_type: Optional[str] = node.value_type

        if isinstance(value_expr, NullLiteral):
            ptr_type = ir.PointerType(ir.IntType(8))
            slot = self.builder.alloca(ptr_type, name=name)
            null_ptr = ir.Constant(ptr_type, None)
            self.builder.store(null_ptr, slot)
            self.env.define(name, slot, ptr_type)
            
            return

        resolved = self.resolve_value(value_expr, value_type)

        if resolved is None:
            raise ValueError(f"Failed to resolve value for variable '{name}'.")

        value_ir, ir_type = resolved

        if self.env.lookup(name) is not None:
            self.report_error(f"Variable '{name}' is already defined in this scope.")
            return
        
        if isinstance(ir_type, ir.PointerType) and isinstance(ir_type.pointee, ir.IdentifiedStructType):  # type: ignore
            # The value is already a struct pointer, so don't wrap it in another alloca
            self.env.define(name, value_ir, ir_type)
            return
            
        if isinstance(ir_type, ir.PointerType) and isinstance(ir_type.pointee, ir.ArrayType):# type: ignore
            zero = ir.Constant(ir.IntType(32), 0)
            elem_ptr = self.builder.gep(value_ir, [zero, zero], inbounds=True, name=f"{name}_elem_ptr")
            slot = self.builder.alloca(elem_ptr.type, name=name)
            self.builder.store(elem_ptr, slot)

            # Store array length for len()
            self.array_lengths[name] = ir_type.pointee.count  # type: ignore

            self.env.define(name, slot, elem_ptr.type)
            return


        ptr = self.builder.alloca(ir_type, name=name)
        self.builder.store(value_ir, ptr)
        self.env.define(name, ptr, ir_type)
       

    def visit_block_statement(self,node:BlockStatement)->None:
        for stmt in node.statements:
            self.compile(stmt)


    def visit_null_literal(self, node: NullLiteral) -> ir.Value:
        null_ptr_type = ir.IntType(32).as_pointer()
        return ir.Constant(null_ptr_type, None)


    def visit_return_statement(self, node: ReturnStatement) -> None:
        if node.return_value is None:
            self.builder.ret_void()
            return 

        expr = node.return_value
        resolved = self.resolve_value(expr)

        if resolved is None:
            raise ValueError("Failed to resolve return expression")

        value, typ = resolved
        self.builder.ret(value)


    def visit_function_statement(self, node: FunctionStatement) -> None:
        if node.name is None or node.name.value is None:
            raise ValueError("Function name is missing or invalid")
        name: str = node.name.value

        if node.body is None:
            raise ValueError(f"Function '{name}' has no body")
        body: BlockStatement = node.body

        params: list[FunctionParameter] = node.parameters or []
        param_names: list[str] = []
        param_types: list[ir.Type] = []

        for param in params:
            if param.name is None:
                raise ValueError(f"Function '{name}' has a parameter with no name")
            param_names.append(param.name)

            if param.value_type in self.type_map:
                param_types.append(self.type_map[param.value_type])
            elif param.value_type == "array":
    
                param_types.append(ir.IntType(32).as_pointer())
            else:
                param_types.append(ir.IntType(32))  
                
   
        func_type = ir.FunctionType(ir.VoidType(), param_types)
        dummy_module = ir.Module(name=f"{name}_dummy_module")
        dummy_func = ir.Function(dummy_module, func_type, name=f"{name}_dummy_temp")
        dummy_block = dummy_func.append_basic_block("entry")


        previous_module = self.module
        previous_builder = self.builder
        previous_env = self.env

        self.module = dummy_module
        self.builder = ir.IRBuilder(dummy_block)
        self.env = Environment(parent=self.env)

        self.env.define(name, dummy_func, ir.VoidType()) 
        
        for i, param_name in enumerate(param_names):
            dummy_ptr = self.builder.alloca(param_types[i])
            self.env.define(param_name, dummy_ptr, param_types[i])

        self.compile(body)
        
        
        if node.return_type is None:
            inferred_type: str | None = None
            for stmt in body.statements:
                if isinstance(stmt, ReturnStatement) and stmt.return_value is not None:
                    result = self.resolve_value(stmt.return_value)
                    if result is not None:
                        _, inferred_ir_type = result
                        if isinstance(inferred_ir_type, ir.IntType) and inferred_ir_type.width == 1:
                            inferred_type = "bool"
                        elif isinstance(inferred_ir_type, ir.IntType):
                            inferred_type = "int"
                        elif isinstance(inferred_ir_type, ir.FloatType):
                            inferred_type = "float"
                        elif isinstance(inferred_ir_type, ir.PointerType):
                            inferred_type = "array"
                        break
            if inferred_type is None:
                inferred_type = "void"
            ret_type_str = inferred_type
            
        else:
            ret_type_str = node.return_type

        node.return_type=ret_type_str
        self.module = previous_module
        self.builder = previous_builder
        self.env = previous_env

        if ret_type_str not in self.type_map:
            if ret_type_str == "array":
                return_ir_type = ir.IntType(32).as_pointer()
            else:
                raise ValueError(f"Unknown return type: {ret_type_str}")
        else:
            return_ir_type = self.type_map[ret_type_str]

    

        func = ir.Function(self.module, ir.FunctionType(return_ir_type, param_types), name=name)


        self.env.define(name, func, return_ir_type)

        block = func.append_basic_block(f'{name}_entry')
        previous_builder = self.builder
        self.builder = ir.IRBuilder(block)

        function_env = Environment(parent=self.env, name=name)
        previous_env = self.env
        self.env = function_env

    
        params_ptr = []
        for i, param_name in enumerate(param_names):
            ptr = self.builder.alloca(param_types[i], name=param_name)
            self.builder.store(func.args[i], ptr)
            self.env.define(param_name, ptr, param_types[i])

        
        

        self.compile(body)

        if ret_type_str == "void" and not any(isinstance(stmt, ReturnStatement) for stmt in body.statements):
            self.builder.ret_void()

    
        self.builder = previous_builder
        self.env=previous_env


    def visit_assign_statement(self, node: AssignStatement) -> None:
    
        if isinstance(node.ident, IdentifierLiteral):
            
            name = node.ident.value
            operator: str = node.operator

            if node.right_value is None:
                self.errors.append(f"COMPILE ERROR: Assignment to '{name}' is missing a right-hand side expression")
                return

           
            result = self.resolve_value(node.right_value)
            if result is None:
                self.errors.append(f"COMPILE ERROR: Cannot resolve right-hand side of assignment to '{name}'")
                return

            right_value, right_type = result

          
            if name is None:
                self.errors.append("COMPILE ERROR: Identifier name is missing")
                return

            entry = self.env.lookup(name)
            if entry is None:
                self.errors.append(f"COMPILE ERROR: Identifier '{name}' has not been declared")
                return

            ptr, var_type = entry
            if ptr is None:
                self.errors.append(f"COMPILE ERROR: No memory pointer for variable '{name}'")
                return
            orig_val = self.builder.load(ptr)
            if orig_val is None:
                self.errors.append(f"COMPILE ERROR: Failed to load value for variable '{name}'")
                return

            
            if isinstance(orig_val.type, ir.FloatType) and isinstance(right_type, ir.IntType):
                right_value = self.builder.sitofp(right_value, ir.FloatType())
            elif isinstance(orig_val.type, ir.IntType) and isinstance(right_type, ir.FloatType):
                orig_val = self.builder.sitofp(orig_val, ir.FloatType())

            value = None
            match operator:
                case '=':
                    value = right_value

                case '+=':
                    if orig_val is not None and isinstance(orig_val.type, ir.IntType):
                        value = self.builder.add(orig_val, right_value)
                    elif orig_val is not None:
                        value = self.builder.fadd(orig_val, right_value)
                    else:
                        self.errors.append("COMPILE ERROR: Cannot perform '+=' on undefined variable")
                        return

                case '-=':
                    if orig_val is not None and isinstance(orig_val.type, ir.IntType):
                        value = self.builder.sub(orig_val, right_value)
                    elif orig_val is not None:
                        value = self.builder.fsub(orig_val, right_value)
                    else:
                        self.errors.append("COMPILE ERROR: Cannot perform '-=' on undefined variable")
                        return

                case '*=':
                    if orig_val is not None and isinstance(orig_val.type, ir.IntType):
                        value = self.builder.mul(orig_val, right_value)
                    elif orig_val is not None:
                        value = self.builder.fmul(orig_val, right_value)
                    else:
                        self.errors.append("COMPILE ERROR: Cannot perform '*=' on undefined variable")
                        return

                case '/=':
                    if orig_val is not None and isinstance(orig_val.type, ir.IntType):
                        value = self.builder.sdiv(orig_val, right_value)
                    elif orig_val is not None:
                        value = self.builder.fdiv(orig_val, right_value)
                    else:
                        self.errors.append("COMPILE ERROR: Cannot perform '/=' on undefined variable")
                        return

                case _:
                    self.errors.append(f"COMPILE ERROR: Unsupported assignment operator '{operator}'")
                    return


            
            self.builder.store(value, ptr)
        
        elif isinstance(node.ident, StructAccessExpression):
            # Handles assignment like 'p.x = 10'
            member_access_node = cast(StructAccessExpression, node.ident)
            member_name = member_access_node.member_name.value
            if member_name is None: return self.report_error("Member assignment missing name.")
            
            # Resolve the struct instance pointer
            obj_resolved = self.resolve_value(member_access_node.struct_name)
            if obj_resolved is None: return self.report_error("Could not resolve object in member assignment.")
            obj_ptr, obj_type = obj_resolved

            if not isinstance(obj_type, ir.PointerType) or not isinstance(obj_type.pointee, ir.IdentifiedStructType): # type: ignore
                return self.report_error("Member assignment requires a struct instance.")

            struct_type = cast(ir.IdentifiedStructType, obj_type.pointee) # type: ignore
            struct_name = struct_type.name
            layout = self.struct_layouts.get(struct_name)
            member_index = layout.get(member_name) if layout else None
            if member_index is None: return self.report_error(f"Struct '{struct_name}' has no member '{member_name}'.")

            # Resolve the value to be stored
            if node.right_value is None: return self.report_error("Assignment is missing a value.")
            right_resolved = self.resolve_value(node.right_value)
            if right_resolved is None: return self.report_error("Could not resolve assignment value.")
            value_to_store, _ = right_resolved

            # Get pointer to the member and store the new value
            zero = ir.Constant(ir.IntType(32), 0)
            member_ptr = self.builder.gep(obj_ptr, [zero, ir.Constant(ir.IntType(32), member_index)], inbounds=True)
            self.builder.store(value_to_store, member_ptr)
            return

        elif isinstance(node.ident, DerefExpression):
            # Resolve the expression inside deref() to get the pointer address.
            deref_node = cast(DerefExpression, node.ident)
            pointer_result = self.resolve_value(deref_node.pointer_expression)
            if pointer_result is None:
                self.report_error("Cannot resolve pointer on the left side of assignment.")
                return

            target_ptr, _ = pointer_result

            # Resolve the right-hand side value that we need to store.
            if node.right_value is None:
                self.report_error("Assignment is missing a right-hand side expression")
                return
            
            right_result = self.resolve_value(node.right_value)
            if right_result is None:
                self.report_error("Cannot resolve right-hand side of assignment.")
                return
            
            value_to_store, _ = right_result

            # Store the new value at the address held by the pointer.
            self.builder.store(value_to_store, target_ptr)
            return


        elif isinstance(node.ident, ArrayAccessExpression):
           
            array_result = self.resolve_value(node.ident.array)
            if array_result is None:
                self.errors.append("COMPILE ERROR: Cannot resolve array in assignment")
                return
            array_val, array_type = array_result

            
            index_result = self.resolve_value(node.ident.index)
            if index_result is None:
                self.errors.append("COMPILE ERROR: Cannot resolve index in array assignment")
                return
            index_val, index_type = index_result

           
            if node.right_value is None:
                self.errors.append("COMPILE ERROR: Assignment to array element missing right-hand side expression")
                return

            right_result = self.resolve_value(node.right_value)
            if right_result is None:
                self.errors.append("COMPILE ERROR: Cannot resolve right-hand side value in array assignment")
                return
            right_value, right_type = right_result

            
            if isinstance(array_type, ir.PointerType) and isinstance(array_type.pointee, ir.ArrayType):# type: ignore
                
                elem_ptr = self.builder.gep(
                    array_val,
                    [ir.Constant(ir.IntType(32), 0), index_val],
                    inbounds=True,
                    name="array_elem_ptr"
                )
                elem_type = array_type.pointee.element# type: ignore
            elif isinstance(array_type, ir.PointerType) and isinstance(array_type.pointee, ir.IntType):# type: ignore
               
                elem_ptr = self.builder.gep(
                    array_val,
                    [index_val],
                    inbounds=True,
                    name="array_elem_ptr"
                )
                elem_type = array_type.pointee# type: ignore
            else:
                self.errors.append("COMPILE ERROR: Cannot index into non-array type in assignment")
                return

        
            if elem_type != right_type:
                if isinstance(elem_type, ir.FloatType) and isinstance(right_type, ir.IntType):
                    right_value = self.builder.sitofp(right_value, elem_type)
                elif isinstance(elem_type, ir.IntType) and isinstance(right_type, ir.FloatType):
                    right_value = self.builder.fptosi(right_value, elem_type)
                else:
                    self.errors.append("COMPILE ERROR: Type mismatch in array assignment")
                    return

           
            self.builder.store(right_value, elem_ptr)

        else:
            self.errors.append("COMPILE ERROR: Left-hand side of assignment must be an identifier or array element")
            return

  
    def visit_break_statement(self, node: BreakStatement) -> None:
        if not self.breakpoints:
            self.report_error("Invalid 'break' used outside of a loop.")
            return

        target_block = self.breakpoints[-1]
        self.builder.branch(target_block)

        dummy_block = self.builder.append_basic_block(name="after_break")
        self.builder.position_at_start(dummy_block)


    def visit_continue_statement(self, node: ContinueStatement) -> None:
        if not self.continues:
            self.report_error("Invalid 'continue' used outside of a loop.")
            return

    
        target_block = self.continues[-1]
        self.builder.branch(target_block)

        dummy_block = self.builder.append_basic_block(name="after_continue")
        self.builder.position_at_start(dummy_block)


    def report_error(self, message: str) -> None:
        self.errors.append(message)
    

    def visit_if_statement(self, node: IfStatement) -> None:
        condition = node.condition
        consequence = node.consequence
        alternative = node.alternative

        if condition is None:
            self.report_error("If condition is missing.")
            return

        result = self.resolve_value(condition)
        if result is None:
            self.report_error("Failed to resolve condition in if-statement")
            return

        test, _ = result

        if alternative is None:
           
            with self.builder.if_then(test):
                if consequence is not None:
                    self.compile(consequence)
        else:
           
            with self.builder.if_else(test) as (then_block, else_block):
                with then_block:
                    if consequence is not None:
                        self.compile(consequence)
                with else_block:
                    if alternative is not None:
                        self.compile(alternative)


    def visit_while_statement(self, node: WhileStatement) -> None:
        condition: Expression = node.condition
        body: BlockStatement = node.body

        while_loop_test = self.builder.append_basic_block(f"while_loop_test_{self.increment_counter()}")
        while_loop_body = self.builder.append_basic_block(f"while_loop_body_{self.counter}")
        while_loop_exit = self.builder.append_basic_block(f"while_loop_exit_{self.counter}")

        self.builder.branch(while_loop_test)

        # Condition evaluation
        self.builder.position_at_start(while_loop_test)
        result = self.resolve_value(condition)
        if result is None:
            self.report_error("Failed to resolve condition in while statement")
            return
        test, _ = result
        self.builder.cbranch(test, while_loop_body, while_loop_exit)

    
        self.builder.position_at_start(while_loop_body)

    
        self.breakpoints.append(while_loop_exit)

        self.compile(body)

        self.breakpoints.pop()

        self.builder.branch(while_loop_test)
        
        self.builder.position_at_start(while_loop_exit)


    def visit_load_statement(self,node:LoadStatement)->None:
        file_path:str=node.file_path

        if not file_path.endswith(".hal"):
            file_path += ".hal"

        if self.global_parsed_pallets.get(file_path) is not None:
            print(f"Hal warns: {file_path} is already imported globally\n")
            return
        

        if file_path in BUILTIN_HEADERS:
            pallet_code = BUILTIN_HEADERS[file_path]
        else:
            search_paths = [
                os.path.abspath(f"headers/{file_path}"),
                os.path.abspath(f"tests/{file_path}")
            ]
            full_path = next((p for p in search_paths if os.path.exists(p)), None)
            if not full_path:
                raise FileNotFoundError(f"Module '{file_path}' not found in built-ins, headers/, or tests/")
            with open(full_path, "r") as f:
                pallet_code = f.read()

        l:Lexer=Lexer(source=pallet_code)
        p:Parser=Parser(lexer=l)
        program:Program=p.parse_program()
        if len(p.errors)>0:
            print(f"error with imported pallet:{file_path}")
            for err in p.errors:
                print(err)
            exit(1)
        self.compile(node=program)
        self.global_parsed_pallets[file_path]=program


    """def visit_while_statement(self,node:WhileStatement)->None:
        condition:Expression=node.condition
        body:BlockStatement=node.body
        result=self.resolve_value(condition)
        if result is None:
            self.report_error("failed to resolve condition in while statment")
            return
        test,_=result
        while_loop_entry=self.builder.append_basic_block(f"while_loop_entry_{self.increment_counter()}")
        while_loop_otherwise=self.builder.append_basic_block(f"while_loop_otherwise_{self.counter}")
        
        self.builder.cbranch(test,while_loop_entry,while_loop_otherwise)
        self.builder.position_at_start(while_loop_entry)
        self.compile(body)
        res=self.resolve_value(condition)
        if res is None:
            self.report_error("failed to resolve condition in while statment")
            return
        test,_=res
        self.builder.cbranch(test,while_loop_entry,while_loop_otherwise)

        self.builder.position_at_start(while_loop_otherwise)"""


    def visit_infix_expression(self, node: InfixExpression) -> tuple[ir.Value, ir.Type] | None:
        operator: str = node.operator

        if node.left_node is None or node.right_node is None:
            self.report_error("InfixExpression is missing left or right operand.")
            return None

        left_result = self.resolve_value(node.left_node)
        right_result = self.resolve_value(node.right_node)
    

        if left_result is None or right_result is None:
            self.report_error("Failed to resolve left or right value.")
            return None
        


        left_value, left_type = left_result
        right_value, right_type = right_result

        value = None
        Type = None

        if node.operator in ("==", "!="):
            # Left is pointer, right is null constant
            if isinstance(left_type, ir.PointerType) and isinstance(right_value, ir.Constant) and right_value.constant is None:
                cmp_val: ir.Instruction = self.builder.icmp_unsigned("==", left_value, right_value, name="null_cmp_eq")
                if cmp_val is None:
                    self.report_error("Failed to build null comparison")
                    return None
                if node.operator == "!=":
                    self.builder.not_(cmp_val, name="null_cmp_neq")
                return cmp_val, ir.IntType(1)

            # Right is pointer, left is null constant
            if isinstance(right_type, ir.PointerType) and isinstance(left_value, ir.Constant) and left_value.constant is None:
                cmp_val: ir.Instruction = self.builder.icmp_unsigned("==", right_value,left_value, name="null_cmp_eq")
                if cmp_val is None:
                    self.report_error("Failed to build null comparison")
                    return None
                if node.operator == "!=":
                    self.builder.not_(cmp_val, name="null_cmp_neq")
                return cmp_val, ir.IntType(1)





        
        
        if isinstance(left_type, ir.IntType) and isinstance(right_type, ir.IntType):
            Type = self.type_map['int']
            match operator:
                case '+':
                    value = self.builder.add(left_value, right_value)
                case '-':
                    value = self.builder.sub(left_value, right_value)
                case '*':
                    value = self.builder.mul(left_value, right_value)
                case '/':
                    value = self.builder.sdiv(left_value, right_value)
                case '%':
                    value = self.builder.srem(left_value, right_value)
                case '^':
                    #TODO:
                    pass
                case '<':
                    value = self.builder.icmp_signed('<',left_value, right_value)
                    Type=ir.IntType(1) 
                case '<=':
                    value = self.builder.icmp_signed('<=',left_value, right_value)
                    Type=ir.IntType(1) 
                case '>':
                    value = self.builder.icmp_signed('>',left_value, right_value)
                    Type=ir.IntType(1) 
                case '>=':
                    value = self.builder.icmp_signed('>=',left_value, right_value)
                    Type=ir.IntType(1) 
                case '==':
                    value = self.builder.icmp_signed('==',left_value, right_value)
                    Type=ir.IntType(1) 
                case '!=':
                    value = self.builder.icmp_signed('!=',left_value, right_value)
                    Type=ir.IntType(1) 
                

        elif isinstance(right_type,ir.FloatType) and isinstance(left_type,ir.FloatType):
            Type=ir.FloatType()
            match operator:
                case '+':
                    value=self.builder.fadd(left_value,right_value)
                case '-':
                    value=self.builder.fsub(left_value,right_value)
                case '*':
                    value=self.builder.fmul(left_value,right_value)
                case '/':
                    value=self.builder.fdiv(left_value,right_value)
                case '%':
                    value=self.builder.frem(left_value,right_value)
                case '^':
                    # TODO
                    pass
                case '<':
                    value = self.builder.fcmp_ordered('<',left_value, right_value)
                    Type=ir.IntType(1) 
                case '<=':
                    value = self.builder.fcmp_ordered('<=',left_value, right_value)
                    Type=ir.IntType(1) 
                case '>':
                    value = self.builder.fcmp_ordered('>',left_value, right_value)
                    Type=ir.IntType(1) 
                case '>=':
                    value = self.builder.fcmp_ordered('>=',left_value, right_value)
                    Type=ir.IntType(1) 
                case '==':
                    value = self.builder.fcmp_ordered('==',left_value, right_value)
                    Type=ir.IntType(1)
                case '!=':
                    value = self.builder.fcmp_ordered('!=',left_value, right_value)
                    Type=ir.IntType(1) 

        if value is not None and Type is not None:
            return value, Type

        self.report_error(f"Unsupported operator '{operator}' or incompatible operand types.")
        return None


    def visit_call_expression(self, node: CallExpression) -> tuple[ir.Value, ir.Type] | None:
        if not isinstance(node.function, IdentifierLiteral) or node.function.value is None:
            self.report_error("CallExpression function must be an identifier with a name.")
            return None
        name: str = node.function.value
        params: list[Expression] = node.arguments if node.arguments is not None else []
        if name in self.struct_types:
            if len(params) != 0:
                self.report_error(f"Struct instantiation for '{name}' does not take arguments.")
                return None
            
            struct_type = self.struct_types[name]
            # `alloca` allocates memory for the struct on the stack and returns a pointer.
            ptr = self.builder.alloca(struct_type, name=f"{name.lower()}_instance")
            return ptr, ir.PointerType(struct_type)

        # ... (rest of the function for handling actual function calls like `print`)
        print(f">>> CallExpression: calling function '{name}' with {len(params)} arguments")
        # ... (Your existing code for print, len, etc., is fine)

        if name == "print":
            args: list[ir.Value] = []
            for i, arg_expr in enumerate(params):
                result = self.resolve_value(arg_expr)
                if result is None:
                    self.report_error(f"Failed to resolve printf argument #{i + 1}")
                    return None
                val, _ = result
                args.append(val)
            ret_ins = self.builtin_printf(params=args, return_type=self.type_map["int"])
            if ret_ins is None:
                return None
            return ret_ins, self.type_map["int"] 
        
        elif name == "len":
            if len(params) != 1:
                self.report_error("len() requires exactly 1 argument.")
                return None

            arg_resolved = self.resolve_value(params[0])
            if arg_resolved is None:
                return None

            _, arg_type = arg_resolved

            # Case 1: Direct fixed-size array
            if isinstance(arg_type, ir.ArrayType):
                return ir.Constant(self.type_map['int'], arg_type.count), self.type_map['int']

            # Case 2: Pointer to fixed-size array
            if isinstance(arg_type, ir.PointerType) and isinstance(arg_type.pointee, ir.ArrayType):  # type: ignore
                return ir.Constant(self.type_map['int'], arg_type.pointee.count), self.type_map['int']  # type: ignore

            # Case 3: Element pointer (from var array declaration)
            if isinstance(arg_type, ir.PointerType) and isinstance(arg_type.pointee, ir.IntType): # type: ignore
                if isinstance(params[0], IdentifierLiteral) and params[0].value is not None:
                    arr_name = params[0].value
                    if arr_name in self.array_lengths:
                        return ir.Constant(self.type_map['int'], self.array_lengths[arr_name]), self.type_map['int']
                self.report_error("len() cannot determine size of pointer without stored length.")
                return None

            self.report_error("len() only works on fixed-size arrays.")
            return None
        


        result = self.env.lookup(name)
        if result is None:
            self.report_error(f"Function '{name}' not found in environment.")
            return None
        assert result is not None  
        func, ret_type = result

     
        func = cast(ir.Function, func)
        expected_arg_types: list[ir.Type] = list(func.function_type.args)
        if len(params) != len(expected_arg_types):
            self.report_error(f"Function '{name}' expects {len(expected_arg_types)} arguments, got {len(params)}.")
            return None

        args: list[ir.Value] = []
        for i, arg_expr in enumerate(params):
            result = self.resolve_value(arg_expr)
            if result is None:
                self.report_error(f"Failed to resolve argument #{i + 1} in call to '{name}'")
                return None
            actual_val, actual_type = result
            print(f">>> Arg #{i+1}: value={actual_val}, type={actual_type}")
            expected_type = expected_arg_types[i]

            

            if actual_type != expected_type:
                if isinstance(expected_type, ir.FloatType) and isinstance(actual_type, ir.IntType):
                    actual_val = self.builder.sitofp(actual_val, expected_type)
                elif isinstance(expected_type, ir.IntType) and isinstance(actual_type, ir.FloatType):
                    actual_val = self.builder.fptosi(actual_val, expected_type)

                elif isinstance(expected_type, ir.PointerType) and isinstance(actual_type, ir.PointerType):
                    zero = ir.Constant(ir.IntType(32), 0)

                
                    if isinstance(actual_type.pointee, ir.ArrayType) and isinstance(expected_type.pointee, ir.IntType): # type: ignore
                        actual_val = self.builder.gep(actual_val, [zero, zero], inbounds=True, name=f"arg{i}_array_to_elem")
                        
                    elif isinstance(expected_type.pointee, ir.ArrayType) and isinstance(actual_type.pointee, ir.IntType):# type: ignore
                        actual_val = self.builder.bitcast(actual_val, expected_type)

                    
                    else:
                        actual_val = self.builder.bitcast(actual_val, expected_type)

                else:
                    self.report_error(f"Type mismatch in arg #{i + 1}: cannot cast {actual_type} to {expected_type}")
                    return None

            
            args.append(actual_val)  # type: ignore


        ret: ir.Value = self.builder.call(func, args)
        
        return ret, ret_type


    def __visit_prefix_expression(self,node:PrefixExpression)->tuple[ir.Value,ir.Type]|None:
        operator:str=node.operator
        if node.right_node is None:
            self.report_error("Prefic Expression is missing left or right operand.")
            return None

        right_node:Expression=node.right_node

        result=self.resolve_value(right_node)
        if result is None:
            raise SyntaxError("fdxhjbfjd")
        right_value,right_type=result
        Type=None
        value=None
        if isinstance(right_type,ir.FloatType):
            Type=ir.FloatType()
            match operator:
                case '-':
                    value=self.builder.fmul(right_value,ir.Constant(ir.FloatType(),-1.0))
        elif isinstance(right_type,ir.IntType):
            Type=ir.IntType(32)
            match operator:
                case '-':
                    value=self.builder.mul(right_value,ir.Constant(ir.IntType(32),-1))

        if value is not None and Type is not None:
            return value, Type

        self.report_error(f"Unsupported operator '{operator}' or incompatible operand types.")
        return None


    def visit_postfix_expression(self, node: PostfixExpression) -> ir.Value | None:
        print(">>> Visiting postfix expression:", type(node.left_node), node.operator)
        if isinstance(node.left_node, IdentifierLiteral):
            print(">>> Postfix on variable:", node.left_node.value)
        if not isinstance(node.left_node, IdentifierLiteral):
            self.report_error("Postfix operator can only be applied to identifiers.")
            return None

        left_node: IdentifierLiteral = node.left_node
        operator: str = node.operator

        if left_node.value is None:
            self.report_error("Identifier in postfix expression has no name.")
            return None

        lookup_result = self.env.lookup(left_node.value)
        if lookup_result is None:
            self.errors.append(f"Compile error: identifier '{left_node.value}' has not been declared.")
            return None

        var_ptr, _ = lookup_result
        orig_value = self.builder.load(var_ptr, name=f"{left_node.value}_val")

        value = None
        match operator:
            case '++':
                if isinstance(orig_value.type, ir.IntType):
                    value = self.builder.add(orig_value, ir.Constant(ir.IntType(32), 1), name=f"{left_node.value}_inc")
                elif isinstance(orig_value.type, ir.FloatType):
                    value = self.builder.fadd(orig_value, ir.Constant(ir.FloatType(), 1.0), name=f"{left_node.value}_finc")
            case '--':
                if isinstance(orig_value.type, ir.IntType):
                    value = self.builder.sub(orig_value, ir.Constant(ir.IntType(32), 1), name=f"{left_node.value}_dec")
                elif isinstance(orig_value.type, ir.FloatType):
                    value = self.builder.fsub(orig_value, ir.Constant(ir.FloatType(), 1.0), name=f"{left_node.value}_fdec")

        if value is not None:
            self.builder.store(value, var_ptr)

  
        return orig_value


    def resolve_value(self, node: Expression, value_type: Optional[str] = None) -> Optional[tuple[ir.Value, ir.Type]]:

        match node.type():
            case NodeType.StructAccessExpression:
                return self.visit_member_access(cast(StructAccessExpression, node))
    
            case NodeType.IntegerLiteral:
                int_node = cast(IntegerLiteral, node)
                value = int_node.value
                Type = self.type_map['int' ]
                return ir.Constant(Type, value), Type

            case NodeType.FloatLiteral:
                float_node = cast(FloatLiteral, node)
                value = float_node.value
                Type = self.type_map['float' ]
                return ir.Constant(Type, value), Type
            
            case NodeType.RefExpression:
                return self.visit_ref_expression(cast(RefExpression, node))

            case NodeType.DerefExpression:
                return self.visit_deref_expression(cast(DerefExpression, node))
            
            case NodeType.ArrayLiteral:
                return self.__visit_array_literal(cast(ArrayLiteral, node))

            case NodeType.ArrayAccessExpression:
                return self.__visit_array_index_expression(cast(ArrayAccessExpression, node))
            
            case NodeType.NullLiteral:
                # Decide LLVM type for null pointer; you can infer or default
                # For now, using i8* (generic pointer) which works for most nullable types
                null_ptr_type = ir.IntType(8).as_pointer()
                llvm_null = ir.Constant(null_ptr_type, None)
                return llvm_null, null_ptr_type

            case NodeType.IdentifierLiteral:
                ident_node = cast(IdentifierLiteral, node)

                if ident_node.value is None:
                    raise ValueError("IdentifierLiteral must have a non-null name")

                result = self.env.lookup(ident_node.value)
                if result is None:
                    self.report_error(f"Undefined variable '{ident_node.value}'")
                    return None

                ptr, typ = result
                if isinstance(typ, ir.PointerType) and isinstance(typ.pointee, ir.ArrayType):  # type: ignore # [N x T]*
                    zero = ir.Constant(ir.IntType(32), 0)
                    if ptr is None:
                        self.report_error(f"Variable '{ident_node.value}' has no memory pointer.")
                        return None
                    if isinstance(ptr.type, ir.PointerType) and isinstance(ptr.type.pointee, ir.PointerType):# type: ignore
               
                        loaded_array_ptr = self.builder.load(ptr, name=f"{ident_node.value}_arrptr_load")
                        elem_ptr = self.builder.gep(loaded_array_ptr, [zero, zero], inbounds=True, name=f"{ident_node.value}_elem_ptr")
                        return elem_ptr, elem_ptr.type
                    else:
                        
                        elem_ptr = self.builder.gep(ptr, [zero, zero], inbounds=True, name=f"{ident_node.value}_elem_ptr")
                        return elem_ptr, elem_ptr.type 
                
                if isinstance(typ, ir.PointerType) and isinstance(typ.pointee, ir.IdentifiedStructType):  # type: ignore
                    return ptr, typ

                # Normal case: load the value
                loaded = self.builder.load(ptr, name=f"{ident_node.value}_load")
                return loaded, loaded.type

            
            case NodeType.BooleanLiteral:
                bool_node = cast(BooleanLiteral, node)
                return ir.Constant(ir.IntType(1), 1 if bool_node.value else 0), ir.IntType(1)

            case NodeType.StringLiteral:
                str_node:StringLiteral=cast(StringLiteral, node)
                if str_node.value is None:
                    self.report_error("String literal has no value.")
                    return None
                string,Type=self.__convert_string(str_node.value)
                return string,Type

            case NodeType.InfixExpression:
                result = self.visit_infix_expression(cast(InfixExpression, node))
                if result is None:
                    self.report_error("Could not resolve value from infix expression.")
                    return None
                return result
            
            case NodeType.PrefixExpression:
                result = self.__visit_prefix_expression(cast(PrefixExpression, node))
                if result is None:
                    self.report_error("Could not resolve value from infix expression.")
                    return None
                return result
            
            case NodeType.PostfixExpression:
                expr = cast(PostfixExpression, node)

                if expr.operator not in ("++", "--"):
                    self.report_error(f"Unsupported postfix operator '{expr.operator}'")
                    return None

                if not isinstance(expr.left_node, IdentifierLiteral):
                    self.report_error("Postfix expressions must target a variable (IdentifierLiteral).")
                    return None

                ident = expr.left_node
                if ident.value is None:
                    self.report_error("PostfixExpression identifier must have a non-null value.")
                    return None
                result = self.env.lookup(ident.value)
                if result is None:
                    self.report_error(f"Variable '{ident.value}' not found.")
                    return None

                ptr, var_type = result
                loaded = self.builder.load(ptr, name=f"{ident.value}_load")

                one = ir.Constant(var_type, 1)
                if expr.operator == "++":
                    updated = self.builder.add(loaded, one, name=f"{ident.value}_inc")
                else:
                    updated = self.builder.sub(loaded, one, name=f"{ident.value}_dec")

                self.builder.store(updated, ptr)
                return loaded, var_type

            case NodeType.InputExpression:
                scanf_func_result = self.env.lookup("input")
                if scanf_func_result is None:
                    self.report_error("Built-in function 'scanf' not found.")
                    return None
                
                scanf_func, _ = scanf_func_result

                format_string = "%d\0"
                c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(format_string)), bytearray(format_string.encode("utf8")))
                global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name=f"__scanf_format_{self.increment_counter()}")
                global_fmt.linkage = 'internal'
                global_fmt.global_constant = True
                global_fmt.initializer = c_fmt # type: ignore
                fmt_ptr = self.builder.bitcast(global_fmt, ir.IntType(8).as_pointer())

                int_ptr = self.builder.alloca(self.type_map['int'], name="input_val_ptr")

                self.builder.call(scanf_func, [fmt_ptr, int_ptr])
                loaded_value = self.builder.load(int_ptr, name="input_val")
                return loaded_value, self.type_map['int']

            case NodeType.CallExpression:
                if isinstance(node, CallExpression):
                    return self.visit_call_expression(node)
                else:
                    self.report_error("Expected CallExpression node, got something else.")
                    return None
                                    
                
    def __visit_array_literal(self, node: ArrayLiteral) -> tuple[ir.Value, ir.Type] | None:
        if not node.elements:
            self.report_error("Array literals must have at least one element.")
            return None

        element_values = []
        element_types = []

        for element_expr in node.elements:
            result = self.resolve_value(element_expr)
            if result is None:
                self.report_error("Failed to resolve one of the array elements.")
                return None
            val, typ = result
            element_values.append(val)
            element_types.append(typ)

        base_type = element_types[0]
        if not all(t == base_type for t in element_types):
            self.report_error("All array elements must have the same type.")
            return None

        array_len = len(element_values)
        array_type = ir.ArrayType(base_type, array_len)
        array_ptr = self.builder.alloca(array_type, name="array")

        for idx, val in enumerate(element_values):
            element_ptr = self.builder.gep(
                array_ptr,
                [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), idx)],
                inbounds=True
            )
            self.builder.store(val, element_ptr)

        return array_ptr, array_ptr.type

    
    def __visit_array_index_expression(self, node: ArrayAccessExpression) -> tuple[ir.Value, ir.Type] | None:
        array_result = self.resolve_value(node.array)
        if array_result is None:
            self.report_error("Failed to resolve array variable in indexing.")
            return None
        array_val, array_type = array_result

        index_result = self.resolve_value(node.index)
        if index_result is None:
            self.report_error("Failed to resolve index in array access.")
            return None
        index_value, index_type = index_result

        if not isinstance(index_type, ir.IntType) or index_type.width != 32:
            self.report_error("Array index must be a 32-bit integer.")
            return None

        # Case A: pointer to an LLVM ArrayType (e.g. [N x i32]*)
        if isinstance(array_type, ir.PointerType) and isinstance(array_type.pointee, ir.ArrayType): # type: ignore
            elem_type = array_type.pointee.element  # type: ignore # e.g. i32
            element_ptr = self.builder.gep(
                array_val,
                [ir.Constant(ir.IntType(32), 0), index_value],
                inbounds=True,
                name="array_element_ptr"
            )
        # Case B: pointer directly to element (e.g. i32*)
        elif isinstance(array_type, ir.PointerType) and isinstance(array_type.pointee, ir.IntType): # type: ignore
            elem_type = array_type.pointee  # type: ignore # e.g. i32
            element_ptr = self.builder.gep(
                array_val,
                [index_value],
                inbounds=True,
                name="array_element_ptr"
            )
        else:
            self.report_error("Cannot index into non-array type.")
            return None

        loaded_value = self.builder.load(element_ptr, name="array_element")
        return loaded_value, elem_type


    def __convert_string(self,string:str)->tuple[ir.GlobalVariable, ir.Type]:
        string=string.replace("\\n","\n\0")
        fmt:str=f"{string}\0"
        c_fmt:ir.Constant=ir.Constant(ir.ArrayType(ir.IntType(8),len(fmt)),bytearray(fmt.encode("utf8")))

        global_fmt=ir.GlobalVariable(self.module,c_fmt.type,name=f'__str_{self.increment_counter()}')
        global_fmt.linkage='internal'
        global_fmt.global_constant=True
        global_fmt.initializer=c_fmt # type: ignore
        return global_fmt,global_fmt.type
    

    def builtin_printf(self, params: list[ir.Value], return_type: ir.Type) -> ir.Instruction | None:
        func_pair = self.env.lookup("print")
        if func_pair is None:
            self.report_error("Built-in function 'printf' not found.")
            return None
        func, _ = func_pair

        if len(params) == 0:
            self.report_error("printf requires at least one argument (the format string).")
            return None

        format_arg = params[0]
        rest_params = []

    
        if not hasattr(self, "true_str"):
            self.true_str = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 5), name="true_str")
            self.true_str.global_constant = True
            self.true_str.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 5), bytearray(b"true\0")) # type: ignore

            self.false_str = ir.GlobalVariable(self.module, ir.ArrayType(ir.IntType(8), 6), name="false_str")
            self.false_str.global_constant = True
            self.false_str.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 6), bytearray(b"false\0")) # type: ignore

        def get_string_ptr(global_var):
            zero = ir.Constant(ir.IntType(32), 0)
            return self.builder.gep(global_var, [zero, zero], inbounds=True)

        
        for val in params[1:]:
            val_type = getattr(val, "type", None)
            if isinstance(val_type, ir.IntType) and val_type.width == 1:
                val = self.builder.select(
                    val,
                    get_string_ptr(self.true_str),
                    get_string_ptr(self.false_str),
                )
            rest_params.append(val)

        
        if isinstance(format_arg, ir.LoadInstr):
            g_var_ptr = format_arg.operands[0]
            string_val = self.builder.load(g_var_ptr)
            fmt_arg = self.builder.bitcast(string_val, ir.IntType(8).as_pointer())
        else:
            fmt_arg = self.builder.bitcast(format_arg, ir.IntType(8).as_pointer())

        return self.builder.call(func, [fmt_arg, *rest_params])


    def visit_ref_expression(self, node: RefExpression) -> tuple[ir.Value, ir.Type] | None:
        if not isinstance(node.expression_to_ref, IdentifierLiteral):
            self.report_error("The 'ref' operator can only be applied to a variable.")
            return None

        name = node.expression_to_ref.value
        if name is None:
            self.report_error("Variable name for 'ref' is missing.")
            return None

        # Look up the variable in the environment. The result is (pointer, type).
        entry = self.env.lookup(name)
        if entry is None:
            self.report_error(f"Cannot take reference of undeclared variable '{name}'.")
            return None

        # The 'ptr' is the memory address allocated for the variable. This is what we want.
        ptr, var_type = entry 
        return ptr, ir.PointerType(var_type)

    def visit_deref_expression(self, node: DerefExpression) -> tuple[ir.Value, ir.Type] | None:
        # First, resolve the inner expression to get the pointer value itself.
        pointer_result = self.resolve_value(node.pointer_expression)
        if pointer_result is None:
            self.report_error("Failed to resolve the pointer expression in deref.")
            return None
        
        pointer_val, pointer_type = pointer_result

        # Check if the resolved value is actually a pointer.
        if not isinstance(pointer_type, ir.PointerType):
            self.report_error("Cannot dereference a non-pointer type.")
            return None

        # Load the value from the memory address the pointer is pointing to.
        loaded_value = self.builder.load(pointer_val, name="deref_load")
        
        # The type of the result is the type the pointer points to (the pointee).
        return loaded_value, pointer_type.pointee # type: ignore
    