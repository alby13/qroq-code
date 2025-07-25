# 🚀 Groq-Code: AI Command Terminal Interface Coding Assistant

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Groq API](https://img.shields.io/badge/Powered%20by-Groq-orange.svg)](https://groq.com/)

A powerful, secure terminal-based (CLI) coding assistant that leverages Groq's lightning-fast AI model (Moonshot Kimi K2) to help you review code, edit files, and get instant coding assistance—all from your command line.

<img src="https://github.com/alby13/qroq-code/blob/main/screenshot1.jpg">

## ✨ Features

### 🤖 **Intelligent AI Assistant**
- **Natural Chat Interface** - Just type your questions and get instant AI responses
- **Code Review** - AI-powered analysis of your code with suggestions and best practices
- **Smart File Editing** - AI assists with code modifications and improvements
- **Context-Aware Responses** - Maintains conversation history for better assistance

### 🛡️ **Enterprise-Grade Security**
- **File Safety Checks** - Automatic binary file detection and size limits
- **Path Validation** - Prevents directory traversal attacks
- **Input Sanitization** - Protects against malicious input
- **Automatic Backups** - Creates backups before file modifications
- **Permission Handling** - Safe file operation with proper error handling

### 🎯 **Developer Experience**
- **Intuitive Navigation** - Tab-based interface with multiple screens
- **Terminal Resize Support** - Handles window resizing gracefully
- **Error Recovery** - Comprehensive error handling with helpful messages
- **Session Persistence** - Saves conversation history and settings
- **Keyboard Shortcuts** - Efficient navigation and control

### ⚡ **Performance Optimized**
- **Groq Integration** - Ultra-fast AI responses powered by Groq's inference engine
- **Rate Limiting** - Smart API usage to prevent quota exhaustion
- **Memory Management** - Efficient handling of large files and conversations
- **Chunked Processing** - Handles large files intelligently

## 🚀 Quick Start

### Prerequisites

- **Python 3.7+** - [Download Python](https://www.python.org/downloads/)
- **Groq API Key** - [Get your free key](https://console.groq.com/keys)
- **Terminal with curses support** - Most Unix/Linux terminals, Windows Terminal, or WSL

### Installation

1. **Clone the repository**
  ```bash
  git clone https://github.com/alby13/groq-code.git
  cd groq-code

Install dependencies
bashpip install -r requirements.txt

Set up your API key (optional - you'll be prompted if not set)
bashexport GROQ_API_KEY="your-groq-api-key-here"

Run the application
bashpython groq_code.py
```

First Run Setup
If you don't have an API key set, Groq-Code will guide you through the setup:
<br>
<br>

## 🔑 GROQ API Key Setup
Your GROQ API key is required to use the AI features.
You can get a free API key at: https://console.groq.com/keys

Set your API key before use:<br>

Command Prompt Windows/DOS:
```
set GROQ_API_KEY=your_actual_groq_api_key_here
```

Linux:
```
export GROQ_API_KEY="your_actual_groq_api_key_here"
```

## 📚 Usage Guide

## Basic Commands
| Command         | Description             | Example                          |
| :-------------- | :---------------------- | :------------------------------- |
| `<message>`     | Chat with AI (default)  | How do I optimize this Python loop? |
| `review <file>` | AI code review          | `review app.py`                  |
| `edit <file> <prompt>` | AI-assisted editing     | `edit main.py add error handling` |
| `help`          | Show help screen        | `help`                           |
| `ls`            | List directory contents | `ls`                             |
| `pwd`           | Show current directory  | `pwd`                            |
| `cd <dir>`      | Change directory        | `cd src`                         |
| `clear`         | Clear current screen    | `clear`                          |
| `exit`          | Quit application        | `exit`                           |

---

### Navigation

| Key      | Action            |
| :------- | :---------------- |
| `Tab`    | Switch between screens |
| `↑`/`↓`  | Scroll content    |
| `PgUp`/`PgDn` | Page up/down      |
| `Ctrl+H` | Command history   |
| `Ctrl+S` | Session statistics |
| `Ctrl+R` | Refresh screen    |
| `Ctrl+Q` | Quit application  |

### Example Workflows

1. Code Review
❯ review myapp.py
📋 Code Review: myapp.py
📊 Size: 2048 bytes
📊 Type: text/x-python
<br>

🤖 Code Review Results:
==================================================
Overall Assessment: Good structure with room for improvement

Issues Found:
1. Missing error handling in line 23
2. Function 'process_data' is too long (50+ lines)
3. No docstrings for public functions

Suggestions:
1. Add try-except blocks for file operations
2. Break down large functions into smaller ones
3. Add type hints for better code clarity
==================================================
2. AI-Assisted Editing
❯ edit utils.py add logging functionality
✏️  Edit Preview for utils.py:
========================================
import logging
import sys

# Configure logging
```
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_logger(name):
    """Set up a logger with the given name."""
    return logging.getLogger(name)
    ... (more content)
========================================
Type 'yes' to apply changes, 'no' to cancel

❯ yes
✅ SUCCESS: File written successfully (backup: utils.py.backup.1703123456)
```

3. Natural Chat
```
❯ What's the difference between list comprehension and generator expressions?

🤖 Great question! Here are the key differences:

**List Comprehensions:**
- Create a complete list in memory: `[x*2 for x in range(10)]`
- Use square brackets `[]`
- Eager evaluation - processes all items immediately
- Higher memory usage for large datasets

**Generator Expressions:**
- Create an iterator object: `(x*2 for x in range(10))`
- Use parentheses `()`
- Lazy evaluation - items generated on-demand
- Memory efficient for large datasets

Use list comprehensions when you need the full list, generators when processing large datasets or when you only need items one at a time.
```

<code>
🏗️ Architecture
Core Components
groq-code/
├── 📁 Core Classes
│   ├── CodingAssistantCLI      # Main application
│   ├── GroqLLMNode            # AI integration
│   ├── FileHandler           # Safe file operations
│   └── CursesErrorHandler    # Terminal management
├── 📁 Safety Systems
│   ├── InputValidator        # Input sanitization
│   ├── SafetyLimits         # Configuration limits
│   └── FileHandler         # File safety checks
└── 📁 UI Components
    ├── Screen              # Display management
    ├── CommandHistory     # Command tracking
    └── StatusBar         # User feedback
</code>

## Security Features

Path Traversal Protection - Validates all file paths
Binary File Detection - Prevents processing of non-text files
Size Limits - Protects against memory exhaustion
Input Sanitization - Filters malicious input
Automatic Backups - Creates backups before modifications
Error Logging - Comprehensive error tracking

## ⚙️ Configuration
You can skip this if you have already set your key earlier, otherwise set your key before use:

Environment Variables
VariableDescriptionDefaultGROQ_API_KEYYour Groq API keyRequiredGROQ_MODELAI model to usemixtral-8x7b-32768

#### Safety Limits
```
pythonMAX_FILE_SIZE_MB = 10        # Maximum file size to process
MAX_CONTENT_LENGTH = 50000   # Maximum content length for display
API_TIMEOUT = 30            # API request timeout in seconds
MIN_TERMINAL_SIZE = (10, 40) # Minimum terminal dimensions
```
### 🔧 Development

#### Requirements
```
txtgroq>=0.4.0
curses-menu>=0.6.0
```

#### Running Tests
bashpython -m pytest tests/
Code Structure
The application follows a modular architecture with clear separation of concerns:

Node-based Processing - Flexible pipeline for AI operations
Error-First Design - Comprehensive error handling at every level
Security-by-Design - Built-in protection against common vulnerabilities
Terminal-Optimized - Efficient rendering and responsive UI

🤝 Contributing
Contributions are welcome! Please see the Contributing Guide (WIP) for details.
Quick Start for Contributors

Fork the repository
Create a feature branch: git checkout -b feature/amazing-feature
Make your changes with tests
Run the test suite: python -m pytest
Submit a pull request

Development Setup
```
bash# Clone your fork
git clone https://github.com/alby13/groq-code.git
cd groq-code

# Install development dependencies
<code>pip install -r requirements-dev.txt</code>

# Run tests
<code>python -m pytest tests/ -v</code>

# Run linting
flake8 groq_code.py
black groq_code.py
```


### 📄 License
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

### 🙏 Acknowledgments

Groq - For providing lightning-fast AI inference
Python curses - Terminal interface foundation
Open Source Community - For inspiration and best practices

### 📞 Support
Getting Help

📖 Documentation: Check this README and inline help (help command)<br>
🐛 Bug Reports: Open an issue<br>
💡 Feature Requests: Start a discussion<br>
💬 Community: Join my Discord server<br>

### Common Issues
<details>
<summary><strong>🔧 Terminal too small error</strong></summary>
Problem: Terminal too small: 20x80 (minimum: 25x80)
Solution: Resize your terminal window or reduce font size to meet minimum requirements.
</details>
<details>
<summary><strong>🔑 API key issues</strong></summary>
Problem: API authentication failures
Solutions:

Verify your API key at Groq Console
Check that GROQ_API_KEY environment variable is set
Try re-running the setup: unset GROQ_API_KEY && python groq_code.py

</details>
<details>
<summary><strong>📁 File permission errors</strong></summary>
Problem: Permission denied when editing files
Solutions:

Check file permissions: ls -la filename
Ensure file isn't open in another application
Run with appropriate permissions if needed

</details>

<div align="center">
  <strong>Built with ❤️ for developers by a developer (alby13 🦊)</strong>
  <br>
  <sub>Powered by Groq's lightning-fast AI inference</sub>
</div>
