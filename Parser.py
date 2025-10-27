from lexer import Lexer
from Token import Token, TokenType
from typing import Callable, Optional, List
from enum import Enum,auto

from AST import Statement,Expression,Program,FunctionStatement,ReturnStatement,BlockStatement,AssignStatement,PostfixExpression,LoadStatement,ArrayLiteral,NullLiteral,StructInstanceExpression
from AST import ExpressionStatement, InfixExpression,IntegerLiteral,FloatLiteral,IdentifierLiteral,VarStatement, PrefixExpression, InputExpression,ArrayAccessExpression,StructStatement,StructAccessExpression,MemberStatement
from AST import BooleanLiteral,IfStatement,CallExpression,FunctionParameter,StringLiteral, WhileStatement,BreakStatement,ContinueStatement,ForStatement,ReserveCall,RefExpression,DerefExpression


class PrecedenceType(Enum):
    P_LOWEST=0
    P_EQUALS=auto()
    P_LESSGREAT=auto()
    P_SUM=auto()
    P_PRODUCT=auto()
    P_EXPONENT=auto()
    P_PREFIX=auto()
    P_CALL=auto()
    P_INDEX=auto()
    TRUE = auto()
    FALSE = auto()
    NULL=auto()



#precendence mapping
PRECEDENCES: dict[TokenType,PrecedenceType]={
    TokenType.PLUS:PrecedenceType.P_SUM,
    TokenType.MINUS:PrecedenceType.P_SUM,
    TokenType.SLASH:PrecedenceType.P_PRODUCT,
    TokenType.ASTERISK:PrecedenceType.P_PRODUCT,
    TokenType.MODULUS:PrecedenceType.P_PRODUCT,
    TokenType.POW:PrecedenceType.P_EXPONENT,
    TokenType.EQ_EQ:PrecedenceType.P_EQUALS,
    TokenType.NOT_EQ:PrecedenceType.P_EQUALS,
    TokenType.LT:PrecedenceType.P_LESSGREAT,
    TokenType.GT:PrecedenceType.P_LESSGREAT,
    TokenType.LT_EQ:PrecedenceType.P_LESSGREAT,
    TokenType.GT_EQ:PrecedenceType.P_LESSGREAT,
    TokenType.LPAREN:PrecedenceType.P_CALL,
    TokenType.LBRACKET: PrecedenceType.P_INDEX,
    TokenType.PLUS_PLUS:PrecedenceType.P_INDEX,
    TokenType.MINUS_MINUS:PrecedenceType.P_INDEX,
    TokenType.DOT: PrecedenceType.P_CALL,
}

class Parser: 
    def __init__(self,lexer:Lexer)->None:
        self.lexer:Lexer=lexer

        self.errors:list[str]=[]

        self.current_token:Optional[Token]= None
        self.peek_token:Optional[Token]=None

        self.prefix_parse_fns: dict[TokenType,Callable]={
            TokenType.IDENTIFIER:self.parse_identifier,
            TokenType.INT:self.parse_int_literal,
            TokenType.FLOAT:self.parse_float_literal,
            TokenType.LPAREN:self.parse_grouped_expression,
            TokenType.IF:self.parse_if_statement,
            TokenType.TRUE:self.parse_boolean,
            TokenType.FALSE:self.parse_boolean,
            TokenType.STRING:self.parse_string_literal,
            TokenType.PLUS: self.parse_prefix_expression,
            TokenType.MINUS: self.parse_prefix_expression,
            TokenType.INPUT: self.parse_input_expression,
            TokenType.LBRACKET: self.parse_array_literal,
            TokenType.NULL:self.parse_null_literal,
            TokenType.RESERVE:self.parse_reserve_call,
            TokenType.FREE:self.parse_identifier,
            TokenType.REF: self.parse_ref_expression,       
            TokenType.DEREF: self.parse_deref_expression,
            TokenType.STRUCT: self.parse_struct_statement,
        } 
        self.infix_parse_fns: dict[TokenType,Callable]={
            TokenType.PLUS:self.parse_infix_expression,
            TokenType.MINUS:self.parse_infix_expression,
            TokenType.SLASH:self.parse_infix_expression,
            TokenType.ASTERISK:self.parse_infix_expression,
            TokenType.MODULUS:self.parse_infix_expression,
            TokenType.POW:self.parse_infix_expression,
            TokenType.NOT_EQ:self.parse_infix_expression,
            TokenType.EQ_EQ:self.parse_infix_expression,
            TokenType.LT:self.parse_infix_expression,
            TokenType.GT:self.parse_infix_expression,
            TokenType.LT_EQ:self.parse_infix_expression,
            TokenType.GT_EQ:self.parse_infix_expression,
            TokenType.LPAREN:self.parse_call_expression,
            TokenType.PLUS_PLUS:self.parse_postfix_expression,
            TokenType.MINUS_MINUS:self.parse_postfix_expression,
            TokenType.LBRACKET:self.parse_array_index_expression,
            TokenType.DOT: self.parse_struct_access_expression,
            
        }

        self.next_token()
        self.next_token()


    def next_token(self) ->None: 
        self.current_token=self.peek_token
        self.peek_token=self.lexer.next_token()

    def current_token_is(self,tt:TokenType) ->bool:
        return self.current_token is not None and self.current_token.type==tt

    def peek_token_is(self, tt: TokenType) -> bool:
        return self.peek_token is not None and self.peek_token.type == tt
    
    def peek_token_is_assignment(self)->bool:
        assignment_operators:list[TokenType]=[
            TokenType.EQ,
            TokenType.PLUS_EQ,
            TokenType.MINUS_EQ,
            TokenType.MUL_EQ,
            TokenType.DIV_EQ

        ]
        if self.peek_token is None:
            return False
        return self.peek_token.type in assignment_operators

    def expect_peek(self,tt:TokenType) ->bool:
        if self.peek_token_is(tt):
            self.next_token()
            return True
        else:
            self.peek_error(tt)
            return False
        
    def current_precedence(self) -> PrecedenceType:
        if self.current_token is None:
            return PrecedenceType.P_LOWEST

        prec: PrecedenceType | None = PRECEDENCES.get(self.current_token.type)
        if prec is None:
            return PrecedenceType.P_LOWEST
        return prec
    
    def __peek_precedence(self) -> PrecedenceType:
        if self.peek_token is None:
            return PrecedenceType.P_LOWEST
        prec: PrecedenceType | None = PRECEDENCES.get(self.peek_token.type)
        if prec is None:
            return PrecedenceType.P_LOWEST
        return prec

    
    def peek_error(self, tt: TokenType) -> None:
        actual = getattr(self.peek_token, "type", "None")
        self.errors.append(f"Expected next token to be {tt}, but got {actual} instead")

    def no_prefix_parse_fn_error(self,tt:TokenType):
        self.errors.append(f"no prefix parsefucntion for {tt} found")




    def parse_program(self) -> Program:
        program: Program = Program()

        while self.current_token is not None and self.current_token.type != TokenType.EOF:
            stmt: Optional[Statement] = self.parse_statement()
            if stmt is not None:
                program.statements.append(stmt)
            if self.current_token is not None and self.current_token.type not in {TokenType.FUN}:
                self.next_token()
        return program
    

    def parse_statement(self) ->Statement|None:
        if self.current_token is None:
            raise SyntaxError("unexpected end of input curret token is none")
        
        if self.current_token.type in {
        TokenType.SEMICOLON,
        TokenType.FI,
        TokenType.ESLE,
        TokenType.ELSE,
        TokenType.NUF,
        }:
            return None 
        
        if self.current_token.type==TokenType.IDENTIFIER and self.peek_token_is_assignment():
            return self.parse_assignment_statement()
        
        """if self.current_token.type == TokenType.IDENTIFIER and self.peek_token_is(TokenType.LBRACKET):
        # Parse the array access first
            array_access = self.parse_expression(PrecedenceType.P_LOWEST)
            if isinstance(array_access, ArrayAccessExpression) and self.peek_token_is_assignment():
                return self.parse_array_element_assignment_statement(array_access)
            else:
                # If it's not followed by '=', treat it as an expression
                return self.parse_expression_statement()"""
        
        match self.current_token.type:
            case TokenType.VAR:
                return self.parse_var_statement()
            case TokenType.FUN:
                return self.parse_function_statement()
            case TokenType.ARRAY:
                return self.parse_array_declaration()
            case TokenType.RETURN:
                return self.parse_return_statement()
            case TokenType.IF:
                return self.parse_if_statement()
            case TokenType.WHILE:
                return self.parse_while_statement()
            #case TokenType.FOR:
             #   return self.parse_for_statement()
            case TokenType.BREAK:
                return self.parse_break_statement()
            case TokenType.CONTINUE:
                return self.parse_continue_statement()
            case TokenType.LOAD:
                return self.parse_load_statement()
                
            case _:
                expr = self.parse_expression(PrecedenceType.P_LOWEST)

                # Check if the *next* token is an assignment operator
                if self.peek_token_is_assignment():
                    self.next_token() # Move to the assignment token
                    return self.parse_assignment_statement(left_expr=expr) # type: ignore

                # Otherwise, it's a regular expression statement
                if self.peek_token_is(TokenType.SEMICOLON):
                    self.next_token()

                # Optional: Prevent standalone literals from being statements
                if isinstance(expr, (IntegerLiteral, FloatLiteral, BooleanLiteral)):
                    return None
                    
                return ExpressionStatement(expr=expr)


    def parse_ref_expression(self) -> Expression:
        if not self.expect_peek(TokenType.LPAREN):
            raise SyntaxError("Expected '(' after 'ref'")

        self.next_token() # Consume the '('
        
        # Parse the identifier inside ref()
        expression_to_ref = self.parse_expression(PrecedenceType.P_LOWEST)

        if not self.expect_peek(TokenType.RPAREN):
            raise SyntaxError("Expected ')' after ref expression")
        
        return RefExpression(expression_to_ref=expression_to_ref)


    def parse_deref_expression(self) -> Expression:
        if not self.expect_peek(TokenType.LPAREN):
            raise SyntaxError("Expected '(' after 'deref'")

        self.next_token() # Consume the '('
        
        # Parse the pointer expression inside deref()
        pointer_expression = self.parse_expression(PrecedenceType.P_LOWEST)

        if not self.expect_peek(TokenType.RPAREN):
            raise SyntaxError("Expected ')' after deref expression")

        return DerefExpression(pointer_expression=pointer_expression)


    def parse_struct_statement(self) -> StructStatement | None:
        if not self.expect_peek(TokenType.IDENTIFIER):
            self.errors.append("Expected struct name after 'struct'")
            return None
        if self.current_token is None:
            self.errors.append("dsdfdf")
            return None
        struct_name = IdentifierLiteral(value=self.current_token.literal)
        
        members: List[MemberStatement] = []
       
        while self.peek_token_is(TokenType.VAR) or self.peek_token_is(TokenType.ARRAY):
            self.next_token()
            

            if self.current_token is None:
                self.errors.append("dsdfdf")
                return None
            
            member_type = None if self.current_token.literal == "var" else self.current_token.literal
            
            if not self.expect_peek(TokenType.IDENTIFIER):
                self.errors.append(f"Expected member name after '{member_type}'")
                return None
            
            member_name = IdentifierLiteral(value=self.current_token.literal)
            
            if not self.expect_peek(TokenType.SEMICOLON):
                self.errors.append("Expected ';' after struct member declaration")
                return None
            
            members.append(MemberStatement(name=member_name, value_type=member_type))
            
    
        if not self.expect_peek(TokenType.TCURTS):
            if self.peek_token is None:
                self.errors.append("dsdfdf")
                return None
            self.errors.append(f"Expected 'tcurts' to close struct declaration, got {self.peek_token.type}")
            return None
            
        
        
        return StructStatement(name=struct_name, members=members)


    def parse_struct_access_expression(self, struct_node: Expression) -> StructAccessExpression | None:
        if not self.expect_peek(TokenType.IDENTIFIER):
            self.errors.append("Expected member name after '.' operator")
            return None
        
        if self.current_token is None:
            self.errors.append("dsdfdf")
            return None
        
        member_name = IdentifierLiteral(value=self.current_token.literal)
        
        return StructAccessExpression(struct_name=struct_node, member_name=member_name)


    def parse_null_literal(self) -> Expression:
        return NullLiteral()


    def parse_reserve_call(self)->ReserveCall:
        if not self.expect_peek(TokenType.LPAREN):
            self.errors.append("Expected '(' after 'reserve'")
            return ReserveCall(IntegerLiteral(0))  # fallback
        
        # Parse size expression
        self.next_token()
        size_expr = self.parse_expression(PrecedenceType.P_LOWEST)
        
        # Expect closing ')'
        if not self.expect_peek(TokenType.RPAREN):
            self.errors.append("Expected ')' after size in reserve()")
            return ReserveCall(size_expr)
        
        return ReserveCall(size_expr)


    def parse_var_statement(self) -> VarStatement:
        stmt: VarStatement = VarStatement()

        if not self.expect_peek(TokenType.IDENTIFIER):
            line = self.current_token.line_no if self.current_token else -1
            raise SyntaxError(f"Expected identifier after 'var' at line {line}")

        if self.current_token is None:
            raise SyntaxError("Expected identifier, but got None")

        stmt.name = IdentifierLiteral(value=self.current_token.literal)

        if self.peek_token_is(TokenType.EQ):
            self.next_token()
            self.next_token()
            stmt.value=self.parse_expression(PrecedenceType.P_LOWEST)
        else:
            stmt.value=None
        if not self.peek_token_is(TokenType.SEMICOLON):
            raise SyntaxError(f"Expected ; after variable declarationa at line {self.current_token.line_no}")
        self.next_token()
        return stmt


    def parse_array_declaration(self) -> VarStatement:
        stmt: VarStatement = VarStatement()
        if self.current_token is None:
            raise SyntaxError("Expected array declaration, but current_token is None")

        if not self.expect_peek(TokenType.IDENTIFIER):
            raise SyntaxError(f"Expected identifier after 'array' at line {self.current_token.line_no}")

        if self.current_token is None:
            raise SyntaxError("Expected identifier, but got None")
        
        stmt.name = IdentifierLiteral(value=self.current_token.literal)

        if not self.expect_peek(TokenType.EQ):
            raise SyntaxError(f"Expected '=' after array identifier at line {self.current_token.line_no}")
        
        self.next_token()
        stmt.value=self.parse_expression(PrecedenceType.P_LOWEST)
        
        if not self.expect_peek(TokenType.SEMICOLON):
            raise SyntaxError(f"Expected ';' after array declaration at line {self.current_token.line_no}")
        
        return stmt

    
    def parse_array_literal(self) -> ArrayLiteral:
        elements: list[Expression] = self.parse_expression_list(TokenType.RBRACKET)
        return ArrayLiteral(elements=elements)


    def parse_array_index_expression(self, array: Expression) -> ArrayAccessExpression:
        self.next_token()
        index = self.parse_expression(PrecedenceType.P_LOWEST)

        if self.current_token is None:
            raise SyntaxError("Expected index expression, but current_token is None")
        if not self.expect_peek(TokenType.RBRACKET):
            raise SyntaxError(f"Expected ']' to close array index expression at line {self.current_token.line_no}")
        
        return ArrayAccessExpression(array=array, index=index)


    def parse_function_statement(self) -> FunctionStatement|None:
        stmt: FunctionStatement = FunctionStatement()

        if not self.expect_peek(TokenType.IDENTIFIER):
            raise SyntaxError("Expected function name after 'fun'")
        if self.current_token is None:
            raise SyntaxError("Expected function name identifier, but got None")

        stmt.name = IdentifierLiteral(value=self.current_token.literal)

        if not self.expect_peek(TokenType.LPAREN):
            raise SyntaxError("Expected '(' after function name")

        stmt.parameters = self.parse_function_parameters() 

    
        self.next_token()  

        stmt.return_type = None
        stmt.body = self.parse_block_statement_until([TokenType.NUF])

        if not self.current_token_is(TokenType.NUF):
            self.errors.append(f"Expected 'nuf' to close function, but got {self.current_token.type}")
            return None
        self.next_token()  

        return stmt


    def parse_function_parameters(self)->list[FunctionParameter]:
        params:list[FunctionParameter]=[]
        if self.peek_token_is(TokenType.RPAREN):
            self.next_token()
            return params
        while True:
            self.next_token()

            if self.current_token is None or self.current_token.type not in {TokenType.VAR, TokenType.ARRAY}:
                raise ValueError("Expected a token, like 'var' or 'array' but got None")
            param_type = self.current_token.literal
            if not self.expect_peek(TokenType.IDENTIFIER):
                raise SyntaxError("Expected identifier after 'var' in function parameter")

            param_name = self.current_token.literal
            param = FunctionParameter(name=param_name, value_type=param_type)

            params.append(param)

            if self.peek_token_is(TokenType.COMMA):
                self.next_token() 
            else:
                break

        if not self.expect_peek(TokenType.RPAREN):
            raise SyntaxError("Expected ')' after function parameter list")

        
        return params


    def parse_return_statement(self)->ReturnStatement:
        stmt:ReturnStatement=ReturnStatement()
        self.next_token()
        
        stmt.return_value=self.parse_expression(PrecedenceType.P_LOWEST)
        if not self.expect_peek(TokenType.SEMICOLON):
            raise SyntaxError("Expected ';' after return value")
        return stmt

#ftfvgbhjnkml,

    def parse_block_statement_until(self, end_tokens: List[TokenType]) -> BlockStatement | None:
        block_stmt = BlockStatement()

        while (
            self.current_token is not None and
            self.current_token.type not in end_tokens and
            self.current_token.type != TokenType.NUF and
            
            self.current_token.type != TokenType.EOF
        ):
            stmt = self.parse_statement()
            if stmt is not None:
                block_stmt.statements.append(stmt)
        
            self.next_token()  

    
        return block_stmt


    def parse_assignment_statement(self, left_expr: Expression | None = None) -> AssignStatement|None:
        if self.current_token is None and left_expr is None:
            raise SyntaxError("Expected identifier or expression, but got None")

        stmt = AssignStatement()

        # if left_expr is passed (e.g. deref(p)), use it directly
        if left_expr is not None:
            stmt.ident = left_expr
        else:
            if self.current_token is None:
                self.errors.append("Invalid assignment: missing token")
                return None

            if self.current_token.literal is None:
                self.errors.append("Invalid assignment: missing identifier literal")
                return None

            stmt.ident = IdentifierLiteral(value=self.current_token.literal)
            self.next_token()

        if self.current_token is None or self.current_token.type not in {
            TokenType.EQ, TokenType.PLUS_EQ, TokenType.MINUS_EQ, TokenType.MUL_EQ, TokenType.DIV_EQ
        }:
            raise SyntaxError("Expected assignment operator after identifier or expression")

        stmt.operator = self.current_token.literal
        self.next_token()

        stmt.right_value = self.parse_expression(PrecedenceType.P_LOWEST)

        if self.peek_token_is(TokenType.SEMICOLON):
            self.next_token()

        return stmt


    def parse_if_statement(self) -> IfStatement | None:
        self.next_token()  
        condition = self.parse_expression(PrecedenceType.P_LOWEST)
        if condition is None:
            self.errors.append("Expected condition after 'if'")
            return None

        consequence = self.parse_block_statement_until([TokenType.ELSE, TokenType.FI])
        if consequence is None:
            self.errors.append("Expected consequence block after 'if' condition")
            return None

        alternative = None


        if self.current_token is not None:
            if self.current_token.type == TokenType.ELSE:
                self.next_token() 

                alternative = self.parse_block_statement_until([TokenType.ESLE])
                if alternative is None:
                    self.errors.append("Expected alternative block after 'else'")
                    return None
                if self.current_token is None or self.current_token.type != TokenType.ESLE:
                    self.errors.append("Expected 'esle' after 'else' block")
                    return None
                self.next_token()  

            
        elif self.current_token is not None and self.current_token.type == TokenType.FI:
            self.next_token()  
        else:
            self.errors.append("Expected 'fi' or 'else' after consequence block")
            return None
        

        return IfStatement(condition, consequence, alternative)


    def parse_while_statement(self) -> WhileStatement | None:
        self.next_token()  
        condition = self.parse_expression(PrecedenceType.P_LOWEST)
        if condition is None:
            self.errors.append("Expected condition after 'while'")
            return None

        body_statements: list[Statement] = []

        while not self.current_token_is(TokenType.ELIHW) and not self.current_token_is(TokenType.EOF):
            stmt = self.parse_statement()
            if stmt is not None:
                body_statements.append(stmt)
            self.next_token()

        return WhileStatement(condition=condition, body=BlockStatement(body_statements))


    def parse_break_statement(self)->BreakStatement:
        self.next_token()
        return BreakStatement()


    def parse_continue_statement(self)->ContinueStatement:
        self.next_token()
        return ContinueStatement()


    """def parse_for_statement(self) -> ForStatement | None:
       
        self.next_token()  # Skip 'for'

        # 🔹 Expect 'var'
        if not self.current_token_is(TokenType.VAR):
            self.errors.append("Expected 'var' after 'for'")
            return None

        var_decl = self.__parse_var_statement()
        if var_decl is None:
            return None

        # 🔹 Must now be SEMICOLON after init
        if not self.current_token_is(TokenType.SEMICOLON):
            self.errors.append("Expected ';' after init in for-loop")
            return None
        self.next_token()  # consume first semicolon

        # 🔹 Parse condition expression
        condition = self.parse_expression(PrecedenceType.P_LOWEST)
        if condition is None:
            return None

        # 🔹 Second semicolon must follow condition
        if not self.current_token_is(TokenType.SEMICOLON):
            self.errors.append("Expected ';' after condition in for-loop")
            return None
        self.next_token()  # consume second semicolon

        # 🔹 Parse update statement (e.g., i = i + 1)
        action = self.__parse_assignment_statement()
        if action is None:
            self.errors.append("Expected update assignment after second ';' in for-loop")
            return None

        # 🔹 Parse body
        body_statements: list[Statement] = []

        while not self.current_token_is(TokenType.ROF) and not self.current_token_is(TokenType.EOF):
            stmt = self.__parse_statement()
            if stmt is not None:
                body_statements.append(stmt)
            else:
                self.next_token()  # skip unexpected token to avoid infinite loop

        if not self.current_token_is(TokenType.ROF):
            self.errors.append("Expected 'rof' to close for-loop")
            return None

        self.next_token()  # consume 'rof'

        return ForStatement(
            var_declaration=var_decl,
            condition=condition,
            action=action,
            body=BlockStatement(body_statements)
        )"""


    def parse_expression_statement(self) -> ExpressionStatement|None:
        expr=self.parse_expression(PrecedenceType.P_LOWEST)
        if expr is None:
            return None

        if isinstance(expr, (IntegerLiteral, FloatLiteral, BooleanLiteral, IdentifierLiteral)):
            return None

        if self.peek_token_is(TokenType.SEMICOLON):
            self.next_token()
        stmt:ExpressionStatement=ExpressionStatement(expr=expr)
        return stmt


    def parse_array_element_assignment_statement(self, array_access: ArrayAccessExpression) -> AssignStatement:
        stmt = AssignStatement()
        stmt.ident = array_access  # store ArrayAccessExpression directly

        self.next_token() 
        if self.current_token is None:
            raise SyntaxError("dssd")
            return  # move past '='
        stmt.operator = self.current_token.literal
        self.next_token()

        stmt.right_value = self.parse_expression(PrecedenceType.P_LOWEST)

        if self.peek_token_is(TokenType.SEMICOLON):
            self.next_token()

        return stmt # consume semicolon


    def parse_expression(self, precedence: PrecedenceType) -> Expression:
        if self.current_token is None:
            self.errors.append("No current token available for prefix parse.")
            return IntegerLiteral(0)

        prefix_fn: Callable | None = self.prefix_parse_fns.get(self.current_token.type)
        if prefix_fn is None:
            self.no_prefix_parse_fn_error(self.current_token.type)
            return IntegerLiteral(0)

        left_expr: Expression = prefix_fn()

        while True:
            if self.peek_token is None:
                break
            if self.peek_token_is(TokenType.SEMICOLON):
                break
            if precedence.value >= self.__peek_precedence().value:
                break

            infix_fn: Callable | None = self.infix_parse_fns.get(self.peek_token.type)
            if infix_fn is None:
                break

            self.next_token()
            left_expr = infix_fn(left_expr)

        return left_expr


    def parse_infix_expression(self, left_node: Expression) -> Expression:
        if self.current_token is None:
            self.errors.append("No current token available during infix parse.")
            return left_node  
        
        infix_expr: InfixExpression = InfixExpression(left_node=left_node,operator=self.current_token.literal)

        precedence = self.current_precedence()
        self.next_token()
        infix_expr.right_node = self.parse_expression(precedence)
        return infix_expr


    def parse_grouped_expression(self) -> Expression:
        self.next_token()
        expr: Expression = self.parse_expression(PrecedenceType.P_LOWEST)

        if not self.expect_peek(TokenType.RPAREN):
            self.errors.append("Expected closing parenthesis ')'")
            return IntegerLiteral(0) 
        return expr


    def parse_call_expression(self,function:Expression)->CallExpression:
        expr:CallExpression=CallExpression(function=function)
        expr.arguments=self.parse_expression_list(TokenType.RPAREN)
        return expr


    def parse_prefix_expression(self) -> PrefixExpression | None:
        if self.current_token is None:
            self.errors.append("Expected token in prefix expression, but got None.")
            return None

        prefix_expr = PrefixExpression(operator=self.current_token.literal)
        self.next_token()

        right = self.parse_expression(PrecedenceType.P_PREFIX)
        if right is None:
            self.errors.append("Expected expression after prefix operator.")
            return None

        prefix_expr.right_node = right
        return prefix_expr


    def parse_postfix_expression(self, left_node: Expression) -> PostfixExpression | None:
        if self.current_token is None:
            self.errors.append("Expected token in postfix expression, but got None.")
            return None

        if self.current_token.type not in (TokenType.PLUS_PLUS, TokenType.MINUS_MINUS):
            self.errors.append(f"Unexpected postfix operator: {self.current_token}")
            return None

        node = PostfixExpression(
            left_node=left_node,
            operator=self.current_token.literal  
        )
        self.next_token()
        return node


    def parse_load_statement(self)->LoadStatement|None:
        if not self.expect_peek(TokenType.STRING):
            return None
        if self.current_token is None:
            self.errors.append("none ")
            return
        stmt: LoadStatement=LoadStatement(file_path=self.current_token.literal)
        if not self.expect_peek(TokenType.SEMICOLON):
            return None
        return stmt


    def parse_expression_list(self,end:TokenType)->list[Expression]:
        e_list:list[Expression]=[]
        if self.peek_token_is(end):
            self.next_token()
            return e_list
        self.next_token()
        e_list.append(self.parse_expression(PrecedenceType.P_LOWEST))
        while self.peek_token_is(TokenType.COMMA):
            self.next_token()
            self.next_token()
            e_list.append(self.parse_expression(PrecedenceType.P_LOWEST))
        if not self.expect_peek(end):
            raise SyntaxError(f"Expected '{end}' to close expression list")
        return e_list


    def parse_identifier(self)->Expression:
        if self.current_token is None:
            raise SyntaxError("Expected identifier, but current_token is None")
        
        ident = IdentifierLiteral(value=self.current_token.literal)

        if self.peek_token_is(TokenType.LBRACKET):
            self.next_token() 
            self.next_token()  
            index_expr = self.parse_expression(PrecedenceType.P_LOWEST)
            if not self.expect_peek(TokenType.RBRACKET):
                raise SyntaxError("Expected ']' after array index")
            
            return ArrayAccessExpression(array=ident, index=index_expr)

        return ident
       

    def parse_int_literal(self) -> Expression:
        int_lit: IntegerLiteral = IntegerLiteral()

        if self.current_token is None:
            self.errors.append("Current token is None while parsing integer.")
            return IntegerLiteral(0)

        try:
            int_lit.value = int(self.current_token.literal)
        except Exception:
            self.errors.append(f"Could not parse `{self.current_token.literal}` as integer.")
            return IntegerLiteral(0)

        return int_lit


    def parse_float_literal(self) -> Expression:
        float_lit: FloatLiteral = FloatLiteral()

        if self.current_token is None:
            self.errors.append("Current token is None while parsing float.")
            return FloatLiteral(0)

        try:
            float_lit.value = float(self.current_token.literal)
        except Exception:
            self.errors.append(f"Could not parse `{self.current_token.literal}` as float.")
            return FloatLiteral(0)

        return float_lit


    def parse_boolean(self)->BooleanLiteral:
        return BooleanLiteral(value=self.current_token_is(TokenType.TRUE))


    def parse_input_expression(self) -> Expression | None:
        if not self.expect_peek(TokenType.LPAREN):
            self.errors.append("Expected '(' after 'input'")
            return None

        if not self.expect_peek(TokenType.RPAREN):
            self.errors.append("Expected ')' to close input() call")
            return None
        
        return InputExpression()

    
    def parse_string_literal(self)->StringLiteral:
        if self.current_token is None:
            raise SyntaxError("Expected string literal, but current_token is None")
        return StringLiteral(value=self.current_token.literal)