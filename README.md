# HalEngine — A High-Level Programming Language

HalEngine is a high-level programming language and compiler designed to explore low-level control, multi-threading, LLVM IR generation, and modern programming features.

## Features

- **Sleep and timing constructs** (`sleep(500ms)` / `sleep(2s)`)
- **Function return type inference**
- **High-level print system** with support for arrays and vectors
- **Fine-grained control flow** (`if ... fi`, `else ... esle`, `while ... elihw`)
- **Support for arrays** (static and dynamic)

## Running HalEngine

You can run HalEngine on Windows using the provided executable:

1. Open a terminal (Command Prompt or PowerShell).
2. Navigate to the HalEngine directory containing `halengine.exe` and create a file with an extension .hal.
3. Run a HalEngine program using:
```bash
halengine.exe path\to\your\program.hal
```
## VS Code Extension

Enhance your HalEngine programming experience with syntax highlighting, code snippets, and debugging support using the official HalEngine VS Code Extension:  
[Download HalEngine VS Code Extension](https://marketplace.visualstudio.com/items?itemName=reshmahegde.halengine)  

*(Or install directly in VS Code by searching for “HalEngine” in the Extensions tab.)*



## Example Programs

### 1. Your first program
```hal
fun main()
   print("Hello World!");
   return 0;
nuf
```

### 1. Basic Computation
```hal
fun main()
    var a = 10;
    var b = 20;
    var c = a + b;
    print("Sum:", c);
    return 0;
nuf





