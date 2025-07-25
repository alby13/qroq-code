# GROQ CODE - AI CODING ASSISTANT v1.0
# Created by alby13, Release Date 7-24-2025
#
# Please remember to set your Groq API Key in your environment first.

import curses
import warnings
import copy
import time
import os
import signal
import stat
import mimetypes
from groq import Groq
import sys
import json
import threading
import itertools
import textwrap
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import traceback
import logging
from pathlib import Path

# Configure logging for debugging
logging.basicConfig(
    filename='coding_assistant_errors.log', 
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SafetyLimits:
    """Configuration for safety limits"""
    MAX_FILE_SIZE_MB = 10  # Maximum file size to process
    MAX_CONTENT_LENGTH = 50000  # Maximum content length for display
    MAX_API_RETRIES = 3
    API_TIMEOUT = 30
    CHUNK_SIZE = 8192  # For reading large files
    MIN_TERMINAL_SIZE = (10, 40)  # Minimum terminal dimensions

class InputValidator:
    """Validates and sanitizes user input"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal attacks"""
        # Remove any path separators and normalize
        filename = os.path.basename(filename)
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\0']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        return filename
    
    @staticmethod
    def validate_file_path(file_path: str) -> Tuple[bool, str]:
        """Validate file path for safety"""
        try:
            # Convert to absolute path to resolve any relative components
            abs_path = os.path.abspath(file_path)
            
            # Check if path contains suspicious patterns
            if '..' in file_path or file_path.startswith('/'):
                return False, "Path traversal detected"
            
            # Check if it's within current working directory or subdirectory
            cwd = os.getcwd()
            if not abs_path.startswith(cwd):
                return False, "Access outside working directory not allowed"
                
            return True, "Valid"
        except Exception as e:
            return False, f"Path validation error: {str(e)}"
    
    @staticmethod
    def validate_command_input(command: str) -> str:
        """Sanitize command input"""
        # Limit command length
        max_length = 1000
        if len(command) > max_length:
            command = command[:max_length]
        
        # Remove null bytes and control characters (except common ones)
        command = ''.join(char for char in command if ord(char) >= 32 or char in ['\t', '\n', '\r'])
        
        return command.strip()

class FileHandler:
    """Safe file operations with comprehensive error handling"""
    
    @staticmethod
    def check_file_safety(file_path: str) -> Tuple[bool, str, Dict]:
        """Comprehensive file safety check"""
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist", {}
            
            # Get file stats
            stat_info = os.stat(file_path)
            file_size = stat_info.st_size
            
            # Check if it's actually a file
            if not os.path.isfile(file_path):
                return False, "Path is not a file", {}
            
            # Check file size
            max_size_bytes = SafetyLimits.MAX_FILE_SIZE_MB * 1024 * 1024
            if file_size > max_size_bytes:
                return False, f"File too large ({file_size / 1024 / 1024:.1f}MB > {SafetyLimits.MAX_FILE_SIZE_MB}MB)", {}
            
            # Check if binary file
            mime_type, _ = mimetypes.guess_type(file_path)
            is_binary = FileHandler.is_binary_file(file_path)
            
            # Check permissions
            readable = os.access(file_path, os.R_OK)
            writable = os.access(file_path, os.W_OK)
            
            info = {
                'size': file_size,
                'mime_type': mime_type,
                'is_binary': is_binary,
                'readable': readable,
                'writable': writable,
                'last_modified': datetime.fromtimestamp(stat_info.st_mtime)
            }
            
            return True, "Safe", info
            
        except PermissionError:
            return False, "Permission denied", {}
        except Exception as e:
            return False, f"File check error: {str(e)}", {}
    
    @staticmethod
    def is_binary_file(file_path: str, chunk_size: int = 1024) -> bool:
        """Detect if file is binary by checking for null bytes"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(chunk_size)
                return b'\0' in chunk
        except Exception:
            return True  # Assume binary if we can't read it
    
    @staticmethod
    def safe_read_file(file_path: str, max_size: Optional[int] = None) -> Tuple[bool, str, str]:
        """Safely read file with size limits and encoding detection"""
        try:
            # Safety check first
            is_safe, message, info = FileHandler.check_file_safety(file_path)
            if not is_safe:
                return False, "", message
            
            if info['is_binary']:
                return False, "", "Cannot read binary file"
            
            # Read with size limit
            max_size = max_size or SafetyLimits.MAX_CONTENT_LENGTH
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_size)
                
                # Check if we hit the limit
                if len(content) == max_size:
                    # Check if there's more content
                    if f.read(1):
                        content += f"\n\n[Content truncated - file larger than {max_size} characters]"
                
                return True, content, "Success"
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1', errors='replace') as f:
                    content = f.read(max_size)
                    return True, content, "Success (latin-1 encoding)"
            except Exception as e:
                return False, "", f"Encoding error: {str(e)}"
        except PermissionError:
            return False, "", "Permission denied"
        except IsADirectoryError:
            return False, "", "Path is a directory, not a file"
        except FileNotFoundError:
            return False, "", "File not found"
        except Exception as e:
            return False, "", f"Read error: {str(e)}"
    
    @staticmethod
    def safe_write_file(file_path: str, content: str, backup: bool = True) -> Tuple[bool, str]:
        """Safely write file with backup option"""
        try:
            # Create backup if file exists and backup is requested
            backup_path = None
            if backup and os.path.exists(file_path):
                backup_path = f"{file_path}.backup.{int(time.time())}"
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                except Exception as e:
                    return False, f"Backup failed: {str(e)}"
            
            # Write content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            success_msg = f"File written successfully"
            if backup_path:
                success_msg += f" (backup: {backup_path})"
            
            return True, success_msg
            
        except PermissionError:
            return False, "Permission denied - cannot write to file"
        except IsADirectoryError:
            return False, "Path is a directory, not a file"
        except OSError as e:
            if e.errno == 28:  # No space left on device
                return False, "No space left on device"
            elif e.errno == 13:  # Permission denied
                return False, "Permission denied"
            else:
                return False, f"OS error: {str(e)}"
        except Exception as e:
            return False, f"Write error: {str(e)}"

class CursesErrorHandler:
    """Handle curses-specific errors and terminal management"""
    
    def __init__(self):
        self.terminal_resized = False
        self.last_size = (0, 0)
        self.min_size = SafetyLimits.MIN_TERMINAL_SIZE
        
    def setup_resize_handler(self):
        """Setup signal handler for terminal resize"""
        def handle_resize(signum, frame):
            self.terminal_resized = True
        
        try:
            signal.signal(signal.SIGWINCH, handle_resize)
        except (AttributeError, OSError):
            # SIGWINCH not available on some systems
            pass
    
    def check_terminal_size(self, stdscr) -> Tuple[bool, str]:
        """Check if terminal size is adequate"""
        try:
            height, width = stdscr.getmaxyx()
            min_height, min_width = self.min_size
            
            if height < min_height or width < min_width:
                return False, f"Terminal too small: {height}x{width} (minimum: {min_height}x{min_width})"
            
            return True, "OK"
        except Exception as e:
            return False, f"Size check error: {str(e)}"
    
    def handle_resize(self, stdscr) -> bool:
        """Handle terminal resize safely"""
        try:
            if not self.terminal_resized:
                return True
                
            # Get new size
            try:
                new_height, new_width = stdscr.getmaxyx()
            except curses.error:
                return False
            
            # Check minimum size
            if new_height < self.min_size[0] or new_width < self.min_size[1]:
                return False
            
            # Only resize if size actually changed
            if (new_height, new_width) != self.last_size:
                try:
                    # Clear and refresh
                    stdscr.clear()
                    
                    # Use resizeterm if available
                    if hasattr(curses, 'resizeterm'):
                        curses.resizeterm(new_height, new_width)
                    
                    # Update size tracking
                    self.last_size = (new_height, new_width)
                    
                    # Consume any pending KEY_RESIZE events
                    stdscr.nodelay(True)
                    while True:
                        try:
                            key = stdscr.getch()
                            if key == -1 or key != curses.KEY_RESIZE:
                                break
                        except curses.error:
                            break
                    stdscr.nodelay(False)
                    
                except curses.error as e:
                    logging.error(f"Resize error: {e}")
                    return False
            
            self.terminal_resized = False
            return True
            
        except Exception as e:
            logging.error(f"Resize handler error: {e}")
            return False
    
    def safe_addstr(self, win, y: int, x: int, text: str, attr=0) -> bool:
        """Safely add string to window with bounds checking"""
        try:
            max_y, max_x = win.getmaxyx()
            
            # Check bounds
            if y >= max_y or x >= max_x or y < 0 or x < 0:
                return False
            
            # Truncate text if necessary
            available_width = max_x - x - 1
            if len(text) > available_width:
                text = text[:available_width]
            
            if text:  # Only add if there's text to add
                win.addstr(y, x, text, attr)
            return True
            
        except curses.error:
            return False
        except Exception as e:
            logging.error(f"addstr error: {e}")
            return False

class BaseNode:
    def __init__(self): self.params,self.successors={},{}
    def set_params(self,params): self.params=params
    def next(self,node,action="default"):
        if action in self.successors: warnings.warn(f"Overwriting successor for action '{action}'")
        self.successors[action]=node; return node
    def prep(self,shared): pass
    def exec(self,prep_res): pass
    def post(self,shared,prep_res,exec_res): pass
    def _exec(self,prep_res): return self.exec(prep_res)
    def _run(self,shared): p=self.prep(shared); e=self._exec(p); return self.post(shared,p,e)
    def run(self,shared): 
        if self.successors: warnings.warn("Node won't run successors. Use Flow.")  
        return self._run(shared)
    def __rshift__(self,other): return self.next(other)
    def __sub__(self,action):
        if isinstance(action,str): return _ConditionalTransition(self,action)
        raise TypeError("Action must be a string")

class _ConditionalTransition:
    def __init__(self,src,action): self.src,self.action=src,action
    def __rshift__(self,tgt): return self.src.next(tgt,self.action)

class Node(BaseNode):
    def __init__(self,max_retries=1,wait=0): super().__init__(); self.max_retries,self.wait=max_retries,wait
    def exec_fallback(self,prep_res,exc): raise exc
    def _exec(self,prep_res):
        for self.cur_retry in range(self.max_retries):
            try: return self.exec(prep_res)
            except Exception as e:
                if self.cur_retry==self.max_retries-1: return self.exec_fallback(prep_res,e)
                if self.wait>0: time.sleep(self.wait)

class GroqLLMNode(Node):
    def __init__(self, model="moonshotai/kimi-k2-instruct", max_retries=3, wait=2, **kwargs):
        super().__init__(max_retries=max_retries, wait=wait, **kwargs)
        self.model = model
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
        # Initialize API client with error handling
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is required.")
            self.client = Groq(api_key=api_key)
        except Exception as e:
            logging.error(f"Groq client initialization failed: {e}")
            raise

    def prep(self, shared):
        """Prepare API request with enhanced safety"""
        try:
            system_prompt = self.params.get("system_prompt", shared.get("system_prompt", "You are a helpful assistant."))
            user_prompt = self.params.get("user_prompt", shared.get("user_prompt", ""))
            history = shared.get("chat_history", [])
            
            # Validate and sanitize prompts
            if not user_prompt.strip():
                raise ValueError("Empty user prompt")
            
            # More conservative token limits
            max_content_length = 6000
            if len(user_prompt) > max_content_length:
                user_prompt = user_prompt[:max_content_length] + "\n[Content truncated due to length]"
            
            # Limit history more aggressively
            limited_history = history[-3:] if len(history) > 3 else history
            
            # Build messages with size checking
            messages = []
            
            # Add limited history
            for msg in limited_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    content = str(msg['content'])[:2000]  # Limit individual message size
                    messages.append({"role": msg['role'], "content": content})
            
            # Add system and user messages
            messages.extend([
                {"role": "system", "content": str(system_prompt)[:1000]},
                {"role": "user", "content": user_prompt}
            ])
            
            return {
                "messages": messages,
                "temperature": max(0.0, min(2.0, self.params.get("temperature", 0.7))),
                "max_tokens": max(50, min(1500, self.params.get("max_tokens", 800))),
            }
        except Exception as e:
            logging.error(f"LLM prep error: {e}")
            raise ValueError(f"Request preparation failed: {str(e)}")

    def exec(self, prep_res):
        """Execute API call with comprehensive error handling"""
        start_time = time.perf_counter()
        
        try:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
            # Make API call with timeout
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prep_res["messages"],
                temperature=prep_res["temperature"],
                max_tokens=prep_res["max_tokens"],
                timeout=SafetyLimits.API_TIMEOUT
            )
            
            duration = time.perf_counter() - start_time
            
            # Validate response
            if not response or not response.choices:
                raise RuntimeError("Invalid API response: no choices")
            
            content = response.choices[0].message.content
            if not content:
                content = "[Empty response from API]"
            
            # Validate usage info
            usage = getattr(response, 'usage', None)
            if usage:
                usage_dict = {
                    "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(usage, 'completion_tokens', 0),
                    "total_tokens": getattr(usage, 'total_tokens', 0)
                }
            else:
                usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            return {
                "content": content,
                "usage": usage_dict,
                "duration": duration
            }
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Groq API error: {error_msg}")
            
            # Classify error types for better handling
            if any(phrase in error_msg.lower() for phrase in ["rate limit", "too many requests"]):
                raise RuntimeError(f"⚠️ Rate limit exceeded. Please wait before trying again.")
            elif any(phrase in error_msg.lower() for phrase in ["timeout", "timed out"]):
                raise RuntimeError(f"⚠️ API timeout. The service may be busy.")
            elif any(phrase in error_msg.lower() for phrase in ["authentication", "unauthorized", "api key"]):
                raise RuntimeError(f"⚠️ Authentication failed. Check your API key.")
            elif any(phrase in error_msg.lower() for phrase in ["network", "connection"]):
                raise RuntimeError(f"⚠️ Network error. Check your internet connection.")
            else:
                raise RuntimeError(f"⚠️ API error: {error_msg}")

    def exec_fallback(self, prep_res, exc):
        """Enhanced fallback with better error messaging"""
        error_msg = str(exc)
        
        # Log detailed error for debugging
        logging.error(f"LLM fallback triggered: {error_msg}", exc_info=True)
        
        # Provide helpful fallback content
        if "rate limit" in error_msg.lower():
            fallback_content = "⚠️ Rate limit reached. Try again in a few moments or use shorter prompts."
        elif "timeout" in error_msg.lower():
            fallback_content = "⚠️ Request timed out. The AI service may be busy."
        elif "authentication" in error_msg.lower():
            fallback_content = "⚠️ Authentication failed. Please check your API key configuration."
        else:
            fallback_content = f"⚠️ AI service temporarily unavailable: {error_msg}"
        
        return {
            "content": fallback_content,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "duration": 0
        }

    def post(self, shared, prep_res, exec_res):
        """Enhanced post-processing with validation"""
        try:
            content = exec_res.get("content", "")
            usage = exec_res.get("usage", {})
            duration = exec_res.get("duration", 0)
            
            # Update shared state safely
            shared["llm_output"] = content
            shared["token_usage"] = usage
            shared["response_speed"] = usage.get("completion_tokens", 0) / max(duration, 0.001)
            
            # Update chat history with validation
            if "chat_history" in shared:
                # Limit chat history size
                if len(shared["chat_history"]) > 50:
                    shared["chat_history"] = shared["chat_history"][-40:]
                
                # Add new messages
                user_msg = prep_res["messages"][-1]["content"] if prep_res["messages"] else ""
                shared["chat_history"].append({"role": "user", "content": user_msg})
                shared["chat_history"].append({"role": "assistant", "content": content})
            
            # Determine result status
            if "⚠️" in content or "error" in content.lower():
                return "error"
            return "success"
            
        except Exception as e:
            logging.error(f"LLM post-processing error: {e}")
            return "error"

class SafeFileReaderNode(Node):
    """Enhanced file reader with comprehensive safety checks"""
    
    def exec(self, prep_res):
        file_path = self.params.get("file_path")
        if not file_path:
            raise ValueError("File path required")
        
        # Validate file path
        is_valid, validation_msg = InputValidator.validate_file_path(file_path)
        if not is_valid:
            raise ValueError(f"Invalid file path: {validation_msg}")
        
        # Safe file read
        success, content, message = FileHandler.safe_read_file(file_path)
        if not success:
            raise RuntimeError(message)
        
        return content

    def post(self, shared, prep_res, exec_res):
        shared["file_content"] = exec_res
        return "success" if exec_res else "empty_file"

class SafeFileWriterNode(Node):
    """Enhanced file writer with backup and safety checks"""
    
    def exec(self, prep_res):
        file_path = self.params.get("file_path")
        new_content = self.params.get("new_content", "")
        backup = self.params.get("backup", True)
        
        if not file_path:
            raise ValueError("File path required")
        
        # Validate file path
        is_valid, validation_msg = InputValidator.validate_file_path(file_path)
        if not is_valid:
            raise ValueError(f"Invalid file path: {validation_msg}")
        
        # Safe file write
        success, message = FileHandler.safe_write_file(file_path, new_content, backup)
        if not success:
            raise RuntimeError(message)
        
        return message

    def post(self, shared, prep_res, exec_res):
        shared["write_result"] = exec_res
        return "success"

class CommandHistory:
    def __init__(self, max_size=100):
        self.commands = []
        self.max_size = max_size
    
    def add(self, command: str, result: str, timestamp: str, tokens_used: int = 0):
        # Sanitize inputs
        command = InputValidator.validate_command_input(command)
        result = str(result)[:500]  # Limit result length
        
        entry = {
            "command": command,
            "result": result,
            "timestamp": timestamp,
            "tokens_used": max(0, tokens_used)
        }
        self.commands.append(entry)
        if len(self.commands) > self.max_size:
            self.commands.pop(0)
    
    def get_recent(self, n=10):
        return self.commands[-n:]
    
    def get_all(self):
        return self.commands

class Screen:
    def __init__(self, name: str, title: str):
        self.name = name
        self.title = title
        self.content = []
        self.scroll_pos = 0
        self.max_content_lines = 1000  # Prevent memory bloat
    
    def add_content(self, content: str):
        # Sanitize content
        if isinstance(content, str):
            # Remove control characters except newlines and tabs
            content = ''.join(char for char in content if ord(char) >= 32 or char in ['\n', '\t', '\r'])
            self.content.append(content)
        
        # Limit content size
        if len(self.content) > self.max_content_lines:
            self.content = self.content[-self.max_content_lines//2:]  # Keep most recent half
            self.scroll_pos = 0
    
    def clear(self):
        self.content = []
        self.scroll_pos = 0

class CodingAssistantCLI:
    def __init__(self, persistence_file="coding_assistant_history.json", model="moonshotai/kimi-k2-instruct"):
        # Initialize error handler first
        self.error_handler = CursesErrorHandler()
        self.error_handler.setup_resize_handler()
        
        # Initialize core state
        self.shared = {
            "chat_history": [], 
            "system_prompt": "You are an expert coding assistant. Provide clear, helpful responses about code review, debugging, and development best practices. Keep responses concise but informative."
        }
        
        # Validate persistence file path
        self.persistence_file = self._validate_persistence_file(persistence_file)
        self.command_history = CommandHistory()

        # Initialize session stats
        self.session_stats = {
            "commands_executed": 0,
            "files_reviewed": 0,
            "files_edited": 0,
            "total_tokens": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "errors_encountered": 0
        }

        # Load persistence with error handling
        self._safe_load_persistence()
        
        # Initialize nodes with error handling
        try:
            self.llm_node = GroqLLMNode(model=model)
            self.file_reader = SafeFileReaderNode()
            self.file_writer = SafeFileWriterNode()
        except Exception as e:
            logging.error(f"Node initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize AI components: {str(e)}")
        
        # UI State
        self.current_screen = "main"
        self.screens = {
            "main": Screen("main", "Coding Assistant - Chat"),
            "history": Screen("history", "Command History"),
            "files": Screen("files", "File Operations"),
            "help": Screen("help", "❓ Help & Commands"),
            "stats": Screen("stats", "Session Statistics")
        }
        
        self.status_message = "Ready - Type 'help' for commands"
        self.input_buffer = ""
        self.pending_confirmation = None
        self.last_error_time = 0
        
        # Initialize help content
        self._init_help_content()

    def _handle_chat_message(self, message: str) -> str:
        """Handle direct chat input without 'chat' command"""
        try:
            if len(message) > 1000:
                message = message[:1000] + "..."
            
            self.shared["user_prompt"] = message
            self.status_message = "AI is thinking..."
            
            # Update command stats
            self.session_stats['commands_executed'] += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            response = self.llm_node.run(self.shared)
            
            # Add to main screen
            self.screens["main"].add_content(f"❯ {message}")
            self.screens["main"].add_content(f"AI {self.shared.get('llm_output', 'No response')}")
            self.screens["main"].add_content("")
            
            tokens = self.shared.get('token_usage', {}).get('total_tokens', 0)
            self.session_stats['total_tokens'] += tokens
            
            result = f"Chat response generated ({tokens} tokens)"
            
            # Add to command history
            self.command_history.add(
                command=f"chat: {message}",
                result=result,
                timestamp=timestamp,
                tokens_used=tokens
            )
            
            self.status_message = result
            
            # Auto-save periodically
            if self.session_stats['commands_executed'] % 5 == 0:
                self.safe_save_persistence()
            
            return result
            
        except Exception as e:
            self.session_stats['errors_encountered'] += 1
            error_msg = f"Chat failed: {str(e)}"
            logging.error(f"Chat message '{message}' failed: {e}", exc_info=True)
            self.display_error(error_msg)
            self.status_message = error_msg
            return error_msg
    
    def _validate_persistence_file(self, persistence_file: str) -> str:
        """Validate persistence file path"""
        try:
            # Sanitize filename
            filename = InputValidator.sanitize_filename(os.path.basename(persistence_file))
            if not filename.endswith('.json'):
                filename += '.json'
            return filename
        except Exception:
            return "coding_assistant_history.json"
    
    def _safe_load_persistence(self):
        """Load persistence with comprehensive error handling"""
        if not os.path.exists(self.persistence_file):
            return
        
        try:
            with open(self.persistence_file, "r", encoding='utf-8') as f:
                data = json.load(f)
                
                # Validate data structure
                if not isinstance(data, dict):
                    logging.warning("Invalid persistence data format")
                    return
                
                # Load chat history with validation
                chat_history = data.get("chat_history", [])
                if isinstance(chat_history, list):
                    # Validate chat history entries
                    valid_history = []
                    for entry in chat_history[-100:]:  # Limit to recent entries
                        if isinstance(entry, dict) and 'role' in entry and 'content' in entry:
                            valid_history.append({
                                'role': str(entry['role'])[:20],
                                'content': str(entry['content'])[:5000]
                            })
                    self.shared["chat_history"] = valid_history
                
                # Load command history with validation
                history_data = data.get("command_history", [])
                if isinstance(history_data, list):
                    for entry in history_data[-50:]:  # Limit to recent entries
                        if isinstance(entry, dict) and all(key in entry for key in ['command', 'result', 'timestamp']):
                            self.command_history.commands.append({
                                'command': str(entry['command'])[:200],
                                'result': str(entry['result'])[:500],
                                'timestamp': str(entry['timestamp']),
                                'tokens_used': max(0, int(entry.get('tokens_used', 0)))
                            })
                
                # Load session stats with validation
                stats_data = data.get("session_stats", {})
                if isinstance(stats_data, dict):
                    for key in ['commands_executed', 'files_reviewed', 'files_edited', 'total_tokens']:
                        if key in stats_data:
                            self.session_stats[key] = max(0, int(stats_data.get(key, 0)))
                
        except json.JSONDecodeError as e:
            logging.error(f"Persistence file corrupted: {e}")
            self._backup_corrupted_file()
        except PermissionError:
            logging.error("Permission denied reading persistence file")
        except Exception as e:
            logging.error(f"Error loading persistence: {e}")
   
    def _backup_corrupted_file(self):
       """Backup corrupted persistence file"""
       try:
           backup_name = f"{self.persistence_file}.corrupted.{int(time.time())}"
           import shutil
           shutil.move(self.persistence_file, backup_name)
           logging.info(f"Corrupted file backed up to: {backup_name}")
       except Exception as e:
           logging.error(f"Failed to backup corrupted file: {e}")
   
    def _init_help_content(self):
       """Initialize help content safely"""
       help_content = [
           "⚙️ CODING ASSISTANT COMMANDS:",
           "",
           "- MAIN COMMANDS:",
           "  review <file>          - AI-powered code review",
           "  edit <file> <prompt>   - Edit/create file with AI",
           "  chat <message>         - General coding discussion",
           "  analyze <file>         - Deep code analysis",
           "",
           "- NAVIGATION:",
           "  TAB                    - Switch between screens",
           "  Ctrl+H                 - View command history",
           "  Ctrl+F                 - File operations",
           "  Ctrl+S                 - Session statistics",
           "  Ctrl+R                 - Refresh current screen",
           "",
           "- FILE OPERATIONS:",
           "  ls                     - List current directory",
           "  pwd                    - Show current directory",
           "  cd <path>              - Change directory",
           "",
           "-  SYSTEM:",
           "  clear                  - Clear current screen",
           "  exit                   - Quit application",
           "  help                   - Show this help",
           "",
           "- SAFETY FEATURES:",
           "  - Automatic file backups before editing",
           "  - Size limits to prevent memory issues",
           "  - Path validation to prevent security issues",
           "  - Binary file detection and protection",
           "",
           "- TIPS:",
           "  - Use arrow keys to scroll through content",
           "  - All conversations are automatically saved",
           "  - File edits show preview before applying",
           "  - Terminal resizing is handled automatically",
           "  - Error logs are saved for debugging"
       ]
       
       for line in help_content:
           self.screens["help"].add_content(line)

    def safe_save_persistence(self):
       """Save persistence with comprehensive error handling"""
       try:
           # Prepare data with size limits
           data = {
               "chat_history": self.shared["chat_history"][-100:],  # Limit size
               "command_history": self.command_history.get_recent(50),
               "session_stats": self.session_stats,
               "app_version": "2.0",
               "saved_at": datetime.now().isoformat()
           }
           
           # Write to temporary file first
           temp_file = f"{self.persistence_file}.tmp"
           try:
               with open(temp_file, "w", encoding='utf-8') as f:
                   json.dump(data, f, indent=2, ensure_ascii=False)
               
               # Atomic move to final location
               import shutil
               shutil.move(temp_file, self.persistence_file)
               
           except Exception as e:
               # Clean up temp file on error
               if os.path.exists(temp_file):
                   try:
                       os.remove(temp_file)
                   except:
                       pass
               raise e
               
       except PermissionError:
           self.status_message = "⚠️ Cannot save session - permission denied"
       except OSError as e:
           if e.errno == 28:  # No space left
               self.status_message = "⚠️ Cannot save session - disk full"
           else:
               self.status_message = f"⚠️ Save error: {str(e)}"
       except Exception as e:
           self.status_message = f"⚠️ Save failed: {str(e)}"
           logging.error(f"Persistence save error: {e}")

    def wrap_text(self, text: str, width: int) -> List[str]:
       """Wrap text safely with error handling"""
       try:
           if not text or width <= 0:
               return [""]
           
           lines = []
           for line in str(text).split('\n'):
               if len(line) <= width:
                   lines.append(line)
               else:
                   wrapped = textwrap.wrap(line, width=max(width-2, 10), 
                                         break_long_words=True, 
                                         break_on_hyphens=True)
                   lines.extend(wrapped if wrapped else [""])
           return lines
       except Exception as e:
           logging.error(f"Text wrapping error: {e}")
           return [str(text)[:width] if text else ""]

    def draw_border(self, win, title: str = ""):
       """Draw border safely"""
       try:
           height, width = win.getmaxyx()
           if height > 2 and width > 2:
               win.box()
               if title and len(title) < width - 4:
                   self.error_handler.safe_addstr(win, 0, 2, f" {title} ", curses.A_BOLD)
       except curses.error:
           pass

    def draw_header(self, stdscr):
       """Draw header with error handling"""
       try:
           height, width = stdscr.getmaxyx()
           
           header_lines = [
               "GROQ CODE - AI CODING ASSISTANT v1.0",
               f"Session: {self.session_stats['session_start']} | Commands: {self.session_stats['commands_executed']} | Tokens: {self.session_stats['total_tokens']} | Errors: {self.session_stats['errors_encountered']}",
               ""
           ]
           
           for i, line in enumerate(header_lines):
               if i < height - 1 and line:
                   x = max(0, (width - len(line)) // 2)
                   self.error_handler.safe_addstr(stdscr, i, x, line, curses.A_BOLD | curses.color_pair(1))
       except Exception as e:
           logging.error(f"Header draw error: {e}")

    def draw_status_bar(self, stdscr):
       """Draw status bar with error handling"""
       try:
           height, width = stdscr.getmaxyx()
           
           # Status content with error indication
           screen_info = f"[{self.current_screen.upper()}]"
           if self.session_stats['errors_encountered'] > 0:
               screen_info += f" ⚠️{self.session_stats['errors_encountered']}"
           
           nav_info = "TAB:Switch | Ctrl+H:History | Ctrl+F:Files | Ctrl+S:Stats | Ctrl+Q:Quit"
           
           left_content = f"{screen_info} {self.status_message}"
           right_content = nav_info
           
           # Truncate if necessary
           if len(left_content) + len(right_content) + 3 > width:
               right_content = "TAB:Switch | Ctrl+Q:Quit"
               if len(left_content) + len(right_content) + 3 > width:
                   left_content = left_content[:width - len(right_content) - 5] + "..."
           
           # Clear and draw status line
           status_line = " " * width
           self.error_handler.safe_addstr(stdscr, height - 1, 0, status_line, curses.A_REVERSE)
           self.error_handler.safe_addstr(stdscr, height - 1, 1, left_content, curses.A_REVERSE | curses.A_BOLD)
           
           right_x = width - len(right_content) - 1
           if right_x > len(left_content) + 2:
               self.error_handler.safe_addstr(stdscr, height - 1, right_x, right_content, curses.A_REVERSE)
               
       except Exception as e:
           logging.error(f"Status bar draw error: {e}")

    def draw_input_bar(self, stdscr):
       """Draw input bar with error handling"""
       try:
           height, width = stdscr.getmaxyx()
           
           prompt = "❯ "
           input_line = f"{prompt}{self.input_buffer}"
           
           # Handle long input
           if len(input_line) > width - 2:
               visible_input = "..." + input_line[-(width - 5):]
           else:
               visible_input = input_line
           
           # Clear and draw input line
           input_bg = " " * width
           self.error_handler.safe_addstr(stdscr, height - 2, 0, input_bg)
           self.error_handler.safe_addstr(stdscr, height - 2, 1, visible_input, curses.A_BOLD)
           
           # Position cursor
           cursor_pos = min(len(visible_input), width - 2)
           try:
               stdscr.move(height - 2, cursor_pos + 1)
           except curses.error:
               pass
               
       except Exception as e:
           logging.error(f"Input bar draw error: {e}")

    def draw_screen_content(self, stdscr):
       """Draw screen content with comprehensive error handling"""
       try:
           height, width = stdscr.getmaxyx()
           screen = self.screens[self.current_screen]
           
           content_start = 4
           content_height = height - 6
           
           if content_height <= 0:
               return  # Terminal too small
           
           # Draw screen title
           title_line = f"═══ {screen.title} ═══"
           if len(title_line) < width - 4:
               self.error_handler.safe_addstr(stdscr, content_start - 1, 2, title_line, 
                                            curses.A_BOLD | curses.color_pair(3))
           
           # Handle empty content
           if not screen.content:
               empty_msg = "No content available. Type 'help' for commands."
               x = max(0, (width - len(empty_msg)) // 2)
               self.error_handler.safe_addstr(stdscr, content_start + content_height // 2, x, 
                                            empty_msg, curses.A_DIM)
               return
           
           # Process content with wrapping
           all_display_lines = []
           for line in screen.content:
               try:
                   wrapped_lines = self.wrap_text(str(line), width - 4)
                   all_display_lines.extend(wrapped_lines)
               except Exception as e:
                   logging.error(f"Content wrapping error: {e}")
                   all_display_lines.append(str(line)[:width-4])
           
           # Calculate scrolling
           total_lines = len(all_display_lines)
           max_scroll = max(0, total_lines - content_height)
           screen.scroll_pos = max(0, min(screen.scroll_pos, max_scroll))
           
           # Display visible lines
           start_idx = screen.scroll_pos
           end_idx = min(start_idx + content_height, total_lines)
           
           for i, line in enumerate(all_display_lines[start_idx:end_idx]):
               y = content_start + i
               if y >= height - 3:
                   break
               
               # Determine color based on content
               color_attr = 0
               if line.startswith("ERROR:") or line.startswith("🚨") or "⚠️" in line:
                   color_attr = curses.color_pair(4)
               elif line.startswith("SUCCESS:") or "✅" in line:
                   color_attr = curses.color_pair(5)
               elif line.startswith("@") or line.startswith("AI:"):
                   color_attr = curses.color_pair(2)
               elif line.startswith("$") or line.startswith("❯") or line.startswith("Command:"):
                   color_attr = curses.color_pair(3)
               
               self.error_handler.safe_addstr(stdscr, y, 2, line, color_attr)
           
           # Draw scroll indicator
           if max_scroll > 0:
               scroll_percent = int((screen.scroll_pos / max_scroll) * 100) if max_scroll > 0 else 0
               scroll_info = f"({screen.scroll_pos + 1}-{min(screen.scroll_pos + content_height, total_lines)}/{total_lines}) {scroll_percent}%"
               if len(scroll_info) < width - 4:
                   self.error_handler.safe_addstr(stdscr, content_start - 1, width - len(scroll_info) - 2, 
                                                scroll_info, curses.A_DIM)
           
       except Exception as e:
           logging.error(f"Screen content draw error: {e}")
           # Try to show error message
           try:
               error_msg = f"Display error: {str(e)[:50]}"
               self.error_handler.safe_addstr(stdscr, height // 2, 2, error_msg, curses.color_pair(4))
           except:
               pass

    def update_history_screen(self):
       """Update history screen safely"""
       try:
           self.screens["history"].clear()
           
           if not self.command_history.commands:
               self.screens["history"].add_content("No commands executed yet.")
               return
           
           self.screens["history"].add_content("Recent Commands:")
           self.screens["history"].add_content("")
           
           for i, entry in enumerate(self.command_history.get_recent(20)):
               try:
                   self.screens["history"].add_content(f"{i+1:2d}. [{entry['timestamp']}]")
                   self.screens["history"].add_content(f"    Command: {entry['command']}")
                   self.screens["history"].add_content(f"    Tokens: {entry['tokens_used']}")
                   result_preview = entry['result'][:100]
                   if len(entry['result']) > 100:
                       result_preview += "..."
                   self.screens["history"].add_content(f"    Result: {result_preview}")
                   self.screens["history"].add_content("")
               except Exception as e:
                   logging.error(f"History entry error: {e}")
                   self.screens["history"].add_content(f"    [Error displaying entry {i+1}]")
                   
       except Exception as e:
           logging.error(f"History screen update error: {e}")
           self.screens["history"].add_content("Error loading command history.")

    def update_stats_screen(self):
       """Update statistics screen safely"""
       try:
           self.screens["stats"].clear()
           
           # Calculate uptime safely
           try:
               start_time = datetime.strptime(self.session_stats['session_start'], "%Y-%m-%d %H:%M:%S")
               uptime = datetime.now() - start_time
               uptime_str = str(uptime).split('.')[0]
           except Exception:
               uptime_str = "Unknown"
           
           stats_info = [
               "📊 SESSION STATISTICS",
               "",
               f"Session Started: {self.session_stats['session_start']}",
               f"Uptime: {uptime_str}",
               f"Commands Executed: {self.session_stats['commands_executed']}",
               f"Files Reviewed: {self.session_stats['files_reviewed']}",
               f"Files Edited: {self.session_stats['files_edited']}",
               f"Total Tokens Used: {self.session_stats['total_tokens']}",
               f"Errors Encountered: {self.session_stats['errors_encountered']}",
               "",
               "💬 CHAT HISTORY",
               f"Total Messages: {len(self.shared['chat_history'])}",
               "",
               "🔧 SYSTEM INFO",
               f"Working Directory: {os.getcwd()}",
               f"Python Version: {sys.version.split()[0]}",
               f"Persistence File: {self.persistence_file}",
               "",
               "🛡️ SAFETY STATUS",
               f"Max File Size: {SafetyLimits.MAX_FILE_SIZE_MB}MB",
               f"API Timeout: {SafetyLimits.API_TIMEOUT}s",
               f"Terminal Min Size: {SafetyLimits.MIN_TERMINAL_SIZE[0]}x{SafetyLimits.MIN_TERMINAL_SIZE[1]}",
           ]
           
           for line in stats_info:
               self.screens["stats"].add_content(line)
               
       except Exception as e:
           logging.error(f"Stats screen update error: {e}")
           self.screens["stats"].add_content("Error loading statistics.")

    def process_command_safely(self, command: str):
       """Process commands with comprehensive error handling"""
       try:
           # Validate and sanitize input
           command = InputValidator.validate_command_input(command)
           if not command:
               return
           
           # Update stats
           self.session_stats['commands_executed'] += 1
           timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           
           # Parse command
           parts = command.strip().split()
           if not parts:
               return
           
           cmd = parts[0].lower()
           args = parts[1:]
           
           # Execute command with error handling
           try:
               result = self._execute_command(cmd, args)
           except Exception as e:
               self.session_stats['errors_encountered'] += 1
               error_msg = f"Command error: {str(e)}"
               logging.error(f"Command '{command}' failed: {e}", exc_info=True)
               result = f"ERROR: {error_msg}"
               self.display_error(error_msg)
           
           # Add to command history
           tokens_used = self.shared.get('token_usage', {}).get('total_tokens', 0)
           self.command_history.add(
               command=command,
               result=result,
               timestamp=timestamp,
               tokens_used=tokens_used
           )
           
           # Update status
           self.status_message = result if len(result) < 100 else result[:97] + "..."
           
           # Auto-save periodically
           if self.session_stats['commands_executed'] % 5 == 0:
               self.safe_save_persistence()
               
       except Exception as e:
           logging.error(f"Command processing error: {e}", exc_info=True)
           self.session_stats['errors_encountered'] += 1
           self.status_message = f"⚠️ Command processing failed: {str(e)}"

    def _execute_command(self, cmd: str, args: List[str]) -> str:
       """Execute individual commands with validation"""
       if cmd == "help":
           self.current_screen = "help"
           return "Help screen displayed"
       
       elif cmd == "clear":
           self.screens[self.current_screen].clear()
           return "Screen cleared"
       
       elif cmd == "exit":
           self.safe_save_persistence()
           sys.exit(0)
       
       elif cmd == "chat":
           return self._handle_chat_command(args)
       
       elif cmd == "review":
           return self._handle_review_command(args)
       
       elif cmd == "edit":
           return self._handle_edit_command(args)
       
       elif cmd in ["yes", "no"] and self.pending_confirmation:
           return self._handle_confirmation(cmd)
       
       elif cmd == "ls":
           return self._handle_ls_command()
       
       elif cmd == "pwd":
           return self._handle_pwd_command()
       
       elif cmd == "cd":
           return self._handle_cd_command(args)
       
       else:
           return f"Unknown command '{cmd}'. Type 'help' for available commands."

    def _handle_chat_command(self, args: List[str]) -> str:
       """Handle chat command safely"""
       if not args:
           return "ERROR: Please provide a message to chat about"
       
       user_message = " ".join(args)
       if len(user_message) > 1000:
           user_message = user_message[:1000] + "..."
       
       self.shared["user_prompt"] = user_message
       self.status_message = "AI is thinking..."
       
       try:
           response = self.llm_node.run(self.shared)
           
           # Add to main screen
           self.screens["main"].add_content(f"❯ {user_message}")
           self.screens["main"].add_content(f"AI {self.shared.get('llm_output', 'No response')}")
           self.screens["main"].add_content("")
           
           tokens = self.shared.get('token_usage', {}).get('total_tokens', 0)
           self.session_stats['total_tokens'] += tokens
           
           return f"Chat response generated ({tokens} tokens)"
           
       except Exception as e:
           error_msg = f"Chat failed: {str(e)}"
           self.screens["main"].add_content(f"🚨 {error_msg}")
           return error_msg

    def _handle_review_command(self, args: List[str]) -> str:
       """Handle review command safely"""
       if not args:
           return "ERROR: Please specify a file to review"
       
       file_path = args[0]
       
       # Validate file path
       is_valid, validation_msg = InputValidator.validate_file_path(file_path)
       if not is_valid:
           return f"ERROR: {validation_msg}"
       
       # Check file safety
       is_safe, safety_msg, file_info = FileHandler.check_file_safety(file_path)
       if not is_safe:
           return f"ERROR: {safety_msg}"
       
       if file_info.get('is_binary'):
           return "ERROR: Cannot review binary files"
       
       try:
           # Read file
           success, content, message = FileHandler.safe_read_file(file_path)
           if not success:
               return f"ERROR: {message}"
           
           # Display file info
           self.screens["main"].add_content(f"📋 Code Review: {file_path}")
           self.screens["main"].add_content(f"📊 Size: {file_info['size']} bytes")
           self.screens["main"].add_content(f"📊 Type: {file_info.get('mime_type', 'text/plain')}")
           
           # Process with AI
           review_prompt = f"""Please review this code file '{file_path}':

{content}

Provide:
1. Overall code quality assessment
2. Potential bugs or issues  
3. Suggestions for improvement
4. Best practices recommendations

Keep your response concise but helpful."""
           
           self.shared["user_prompt"] = review_prompt
           self.status_message = "AI is reviewing your code..."
           
           response = self.llm_node.run(self.shared)
           
           self.screens["main"].add_content("=" * 50)
           self.screens["main"].add_content(f"AI {self.shared.get('llm_output', 'No response')}")
           self.screens["main"].add_content("=" * 50)
           self.screens["main"].add_content("")
           
           tokens = self.shared.get('token_usage', {}).get('total_tokens', 0)
           self.session_stats['total_tokens'] += tokens
           self.session_stats['files_reviewed'] += 1
           
           return f"Code review completed for {file_path} ({tokens} tokens)"
           
       except Exception as e:
           error_msg = f"Review failed: {str(e)}"
           self.screens["main"].add_content(f"🚨 {error_msg}")
           return error_msg
   
    def _clean_ai_code_output(self, content: str) -> str:
        """Remove markdown code blocks from AI output"""
        lines = content.strip().split('\n')
        
        # Remove opening code block (```python, ```javascript, etc.)
        if lines and lines[0].strip().startswith('```'):
            lines = lines[1:]
        
        # Remove closing code block
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        
        return '\n'.join(lines)

    def _handle_edit_command(self, args: List[str]) -> str:
       """Handle edit command safely"""
       if len(args) < 2:
           return "ERROR: Usage: edit <file> <prompt>"
       
       file_path = args[0]
       edit_prompt = " ".join(args[1:])
       
       # Validate file path
       is_valid, validation_msg = InputValidator.validate_file_path(file_path)
       if not is_valid:
           return f"ERROR: {validation_msg}"
       
       try:
           # Read existing file if it exists
           existing_content = ""
           if os.path.exists(file_path):
               success, existing_content, message = FileHandler.safe_read_file(file_path)
               if not success:
                   return f"ERROR: Cannot read existing file: {message}"
           
           # AI edit request
           ai_prompt = f"""Edit this file according to the prompt. 

    Current content:
    {existing_content}

    Edit request: {edit_prompt}

    Please provide the complete updated file content without markdown code blocks."""
           
           self.shared["user_prompt"] = ai_prompt
           self.status_message = "AI is editing your file..."
           
           response = self.llm_node.run(self.shared)

           # Clean the AI output to remove markdown code blocks
           raw_content = self.shared.get('llm_output', '')
           new_content = self._clean_ai_code_output(raw_content)
           
           # Show preview
           new_content = self.shared.get('llm_output', '')
           self.screens["main"].add_content(f"✏️  Edit Preview for {file_path}:")
           self.screens["main"].add_content("=" * 40)
           
           # Show first few lines of the edit
           preview_lines = new_content.split('\n')[:10]
           for line in preview_lines:
               self.screens["main"].add_content(f"  {line}")
           
           if len(new_content.split('\n')) > 10:
               self.screens["main"].add_content("  ... (more content)")
           
           self.screens["main"].add_content("=" * 40)
           self.screens["main"].add_content("Type 'yes' to apply changes, 'no' to cancel")
           self.screens["main"].add_content("")
           
           self.pending_confirmation = {
               "type": "edit",
               "file_path": file_path,
               "content": new_content
           }
           
           tokens = self.shared.get('token_usage', {}).get('total_tokens', 0)
           self.session_stats['total_tokens'] += tokens
           
           return f"Edit preview generated for {file_path} ({tokens} tokens)"
           
       except Exception as e:
           error_msg = f"Edit failed: {str(e)}"
           self.screens["main"].add_content(f"🚨 {error_msg}")
           return error_msg

    def _handle_confirmation(self, cmd: str) -> str:
       """Handle confirmation commands safely"""
       if cmd == "yes" and self.pending_confirmation:
           if self.pending_confirmation["type"] == "edit":
               try:
                   file_path = self.pending_confirmation["file_path"]
                   content = self.pending_confirmation["content"]
                   
                   success, message = FileHandler.safe_write_file(file_path, content, backup=True)
                   
                   if success:
                       self.session_stats['files_edited'] += 1
                       result = f"✅ SUCCESS: {message}"
                   else:
                       result = f"ERROR: {message}"
                   
                   self.pending_confirmation = None
                   return result
                   
               except Exception as e:
                   self.pending_confirmation = None
                   return f"ERROR: Write failed: {str(e)}"
       
       elif cmd == "no":
           self.pending_confirmation = None
           return "Operation cancelled"
       
       return "No pending confirmation"

    def _handle_ls_command(self) -> str:
       """Handle ls command safely"""
       try:
           items = []
           for item in os.listdir('.'):
               try:
                   if os.path.isdir(item):
                       items.append(f"📁 {item}/")
                   else:
                       size = os.path.getsize(item)
                       if size > 1024*1024:
                           size_str = f"{size/1024/1024:.1f}MB"
                       elif size > 1024:
                           size_str = f"{size/1024:.1f}KB"
                       else:
                           size_str = f"{size}B"
                       items.append(f"📄 {item} ({size_str})")
               except (PermissionError, OSError):
                   items.append(f"❓ {item} (access denied)")
           
           self.screens["main"].add_content("📁 Current Directory Contents:")
           for item in sorted(items):
               self.screens["main"].add_content(f"  {item}")
           self.screens["main"].add_content("")
           
           return f"Listed {len(items)} items"
           
       except PermissionError:
           return "ERROR: Permission denied"
       except Exception as e:
           return f"ERROR: {str(e)}"

    def _handle_pwd_command(self) -> str:
       """Handle pwd command safely"""
       try:
           cwd = os.getcwd()
           self.screens["main"].add_content(f"📍 Current Directory: {cwd}")
           return f"Current directory: {cwd}"
       except Exception as e:
           return f"ERROR: {str(e)}"

    def _handle_cd_command(self, args: List[str]) -> str:
       """Handle cd command safely"""
       if not args:
           return "ERROR: Please specify a directory"
       
       target_dir = args[0]
       
       # Basic path validation
       if '..' in target_dir or target_dir.startswith('/'):
           return "ERROR: Path traversal not allowed"
       
       try:
           # Check if directory exists and is accessible
           if not os.path.exists(target_dir):
               return f"ERROR: Directory '{target_dir}' does not exist"
           
           if not os.path.isdir(target_dir):
               return f"ERROR: '{target_dir}' is not a directory"
           
           if not os.access(target_dir, os.R_OK | os.X_OK):
               return "ERROR: Permission denied"
           
           # Change directory
           os.chdir(target_dir)
           new_dir = os.getcwd()
           self.screens["main"].add_content(f"📍 Changed to: {new_dir}")
           return f"Changed directory to {new_dir}"
           
       except PermissionError:
           return "ERROR: Permission denied"
       except Exception as e:
           return f"ERROR: {str(e)}"

    def display_error(self, error_msg: str):
       """Display error with helpful information"""
       try:
           current_time = time.time()
           
           # Rate limit error display
           if current_time - self.last_error_time < 1.0:
               return
           
           self.last_error_time = current_time
           
           self.screens["main"].add_content("🚨 ERROR:")
           self.screens["main"].add_content(f"   {error_msg}")
           self.screens["main"].add_content("")
           
           # Add troubleshooting tips for specific errors
           if "API" in error_msg:
               self.screens["main"].add_content("💡 Troubleshooting tips:")
               self.screens["main"].add_content("   • Check your GROQ_API_KEY environment variable")
               self.screens["main"].add_content("   • Verify your Groq API key is valid")
               self.screens["main"].add_content("   • Try again in a few moments (rate limiting)")
               self.screens["main"].add_content("   • Check your internet connection")
               self.screens["main"].add_content("")
           elif "Permission" in error_msg:
               self.screens["main"].add_content("💡 File permission tips:")
               self.screens["main"].add_content("   • Check if file is open in another application")
               self.screens["main"].add_content("   • Verify you have read/write permissions")
               self.screens["main"].add_content("   • Try running with appropriate permissions")
               self.screens["main"].add_content("")
           elif "File" in error_msg and "not found" in error_msg:
               self.screens["main"].add_content("💡 File access tips:")
               self.screens["main"].add_content("   • Check file path spelling")
               self.screens["main"].add_content("   • Use 'ls' to see available files")
               self.screens["main"].add_content("   • Use 'pwd' to check current directory")
               self.screens["main"].add_content("")
       except Exception as e:
           logging.error(f"Error display failed: {e}")

    def handle_input_safely(self, key):
        """Handle keyboard input with comprehensive error handling"""
        try:
            # Handle terminal resize first
            if key == curses.KEY_RESIZE:
                self.error_handler.terminal_resized = True
                return
            
            if key == ord('\n') or key == 10:  # Enter
                if self.input_buffer.strip():
                    # Check if input starts with a known command
                    parts = self.input_buffer.strip().split()
                    if parts:
                        cmd = parts[0].lower()
                        known_commands = ['help', 'clear', 'exit', 'review', 'edit', 'yes', 'no', 'ls', 'pwd', 'cd']
                        
                        if cmd in known_commands:
                            # Process as command
                            self.process_command_safely(self.input_buffer.strip())
                        else:
                            # Process as chat message
                            self._handle_chat_message(self.input_buffer.strip())
                    
                    self.input_buffer = ""
            
            elif key == ord('\t'):  # Tab - switch screens
                screens = list(self.screens.keys())
                current_idx = screens.index(self.current_screen)
                self.current_screen = screens[(current_idx + 1) % len(screens)]
                
                # Update screen-specific content safely
                try:
                    if self.current_screen == "history":
                        self.update_history_screen()
                    elif self.current_screen == "stats":
                        self.update_stats_screen()
                except Exception as e:
                    logging.error(f"Screen update error: {e}")
                    self.status_message = f"⚠️ Screen update failed: {str(e)}"
            
            elif key in [8, 127, curses.KEY_BACKSPACE]:  # Backspace
                if self.input_buffer:
                    self.input_buffer = self.input_buffer[:-1]
            
            elif key == 18:  # Ctrl+R - refresh
                try:
                    if self.current_screen == "history":
                        self.update_history_screen()
                    elif self.current_screen == "stats":
                        self.update_stats_screen()
                    self.status_message = "Screen refreshed"
                except Exception as e:
                    self.status_message = f"⚠️ Refresh failed: {str(e)}"
            
            elif key == 17:  # Ctrl+Q - quit
                self.safe_save_persistence()
                sys.exit(0)
            
            elif key == 8:  # Ctrl+H - history
                self.current_screen = "history"
                try:
                    self.update_history_screen()
                except Exception as e:
                    self.status_message = f"⚠️ History load failed: {str(e)}"
            
            elif key == 6:  # Ctrl+F - files
                self.current_screen = "files"
            
            elif key == 19:  # Ctrl+S - stats
                self.current_screen = "stats"
                try:
                    self.update_stats_screen()
                except Exception as e:
                    self.status_message = f"⚠️ Stats load failed: {str(e)}"
            
            elif key == curses.KEY_UP:  # Scroll up
                screen = self.screens[self.current_screen]
                if screen.scroll_pos > 0:
                    screen.scroll_pos -= 1
            
            elif key == curses.KEY_DOWN:  # Scroll down
                try:
                    screen = self.screens[self.current_screen]
                    if screen.content:
                        # Calculate max scroll safely
                        height, width = 25, 80  # Default values
                        content_height = height - 6
                        
                        all_display_lines = []
                        for line in screen.content:
                            wrapped_lines = self.wrap_text(str(line), width - 4)
                            all_display_lines.extend(wrapped_lines)
                        
                        max_scroll = max(0, len(all_display_lines) - content_height)
                        if screen.scroll_pos < max_scroll:
                            screen.scroll_pos += 1
                except Exception as e:
                    logging.error(f"Scroll calculation error: {e}")
            
            elif key == curses.KEY_PPAGE:  # Page Up
                screen = self.screens[self.current_screen]
                screen.scroll_pos = max(0, screen.scroll_pos - 10)
            
            elif key == curses.KEY_NPAGE:  # Page Down
                try:
                    screen = self.screens[self.current_screen]
                    if screen.content:
                        height, width = 25, 80
                        content_height = height - 6
                        
                        all_display_lines = []
                        for line in screen.content:
                            wrapped_lines = self.wrap_text(str(line), width - 4)
                            all_display_lines.extend(wrapped_lines)
                        
                        max_scroll = max(0, len(all_display_lines) - content_height)
                        screen.scroll_pos = min(max_scroll, screen.scroll_pos + 10)
                except Exception as e:
                    logging.error(f"Page scroll error: {e}")
            
            elif 32 <= key <= 126:  # Printable characters
                # Limit input buffer size
                if len(self.input_buffer) < 500:
                    self.input_buffer += chr(key)
            
        except Exception as e:
            logging.error(f"Input handling error: {e}")
            self.session_stats['errors_encountered'] += 1
            self.status_message = f"⚠️ Input error: {str(e)}"

    def init_colors_safely(self):
       """Initialize colors with error handling"""
       try:
           if not curses.has_colors():
               return
           
           curses.start_color()
           
           # Define colors safely
           if curses.can_change_color():
               try:
                   curses.init_color(curses.COLOR_WHITE, 900, 900, 900)
                   
                   # Define custom colors if possible
                   ORANGE = 8
                   BRIGHT_WHITE = 9
                   curses.init_color(ORANGE, 800, 400, 0)
                   curses.init_color(BRIGHT_WHITE, 1000, 1000, 1000)
               except curses.error:
                   # Fallback to standard colors
                   ORANGE = curses.COLOR_YELLOW
                   BRIGHT_WHITE = curses.COLOR_WHITE
           else:
               ORANGE = curses.COLOR_YELLOW
               BRIGHT_WHITE = curses.COLOR_WHITE
           
           # Initialize color pairs safely
           color_pairs = [
               (1, BRIGHT_WHITE, curses.COLOR_BLACK),    # Header
               (2, curses.COLOR_GREEN, curses.COLOR_BLACK),   # AI responses  
               (3, ORANGE, curses.COLOR_BLACK),          # Commands
               (4, curses.COLOR_RED, curses.COLOR_BLACK),     # Errors
               (5, curses.COLOR_GREEN, curses.COLOR_BLACK),   # Success
               (6, curses.COLOR_CYAN, curses.COLOR_BLACK),    # Info
           ]
           
           for pair_num, fg, bg in color_pairs:
               try:
                   curses.init_pair(pair_num, fg, bg)
               except curses.error:
                   # Skip if color pair initialization fails
                   pass
                   
       except Exception as e:
           logging.error(f"Color initialization error: {e}")
           # Continue without colors

    def draw_splash_screen_safely(self, stdscr):
       """Draw splash screen with error handling"""
       try:
           height, width = stdscr.getmaxyx()
           stdscr.clear()
           
           splash_art = [
            "╔══════════════════════════════════════════════════════╗",
            "║                                                      ║",
            "║          ██████ ██████   ██████   ██████             ║",
            "║         ██      ██   ██ ██    ██ ██    ██            ║",
            "║         ██   ██ ██  ███ ██    ██ ██    ██            ║",
            "║         ██    █ ████    ██    ██ ██  █ ██            ║",
            "║          ██████ ██  ███  ██████   ███ █              ║",
            "║                                                      ║",
            "║          AI-POWERED CODING ASSISTANT v1.0            ║",
            "║                                                      ║",
            "║              Welcome to your personal                ║",
            "║           intelligent coding companion!              ║",
            "║                                                      ║",
            "║   Features:                                          ║",
            "║   • Smart code review and analysis                   ║",
            "║   • AI-powered file editing                          ║",
            "║   • Interactive coding discussions                   ║",
            "║   • Persistent session history                       ║",
            "║                                                      ║",
            "║           Press any key to continue...               ║",
            "║                                                      ║",
            "╚══════════════════════════════════════════════════════╝"
           ]


           
           start_y = max(0, (height - len(splash_art)) // 2)
           
           for i, line in enumerate(splash_art):
               y = start_y + i
               if y < height - 1:
                   x = max(0, (width - len(line)) // 2)
                   try:
                       if "█" in line:
                           attr = curses.A_BOLD | curses.color_pair(3)
                       elif "Features:" in line:
                           attr = curses.A_BOLD | curses.color_pair(3)
                       elif "Press" in line:
                           attr = curses.A_BOLD | curses.color_pair(3)
                       else:
                           attr = curses.color_pair(1)
                       
                       self.error_handler.safe_addstr(stdscr, y, x, line, attr)
                   except curses.error:
                       pass
           
           stdscr.refresh()
           
           # Wait for keypress with timeout
           stdscr.timeout(5000)  # 5 second timeout
           key = stdscr.getch()
           stdscr.timeout(-1)  # Reset to blocking
           
       except Exception as e:
           logging.error(f"Splash screen error: {e}")
           # Continue anyway

    def run_safely(self):
       """Main application loop with comprehensive error handling"""
       def main_loop(stdscr):
           try:
               # Initialize curses safely
               try:
                   curses.curs_set(0)
                   stdscr.keypad(True)
                   stdscr.timeout(-1)  # Blocking mode
                   self.init_colors_safely()
               except curses.error as e:
                   logging.error(f"Curses initialization error: {e}")
                   # Continue with limited functionality
               
               # Check initial terminal size
               is_adequate, size_msg = self.error_handler.check_terminal_size(stdscr)
               if not is_adequate:
                   try:
                       stdscr.clear()
                       error_msg = f"Terminal too small: {size_msg}"
                       self.error_handler.safe_addstr(stdscr, 0, 0, error_msg, curses.A_BOLD)
                       self.error_handler.safe_addstr(stdscr, 1, 0, "Please resize terminal and restart", curses.A_BOLD)
                       stdscr.refresh()
                       stdscr.getch()
                       return
                   except curses.error:
                       return
               
               # Show splash screen
               try:
                   self.draw_splash_screen_safely(stdscr)
               except Exception as e:
                   logging.error(f"Splash screen failed: {e}")
               
               # Main application loop
               consecutive_errors = 0
               max_consecutive_errors = 10
               
               while True:
                   try:
                       # Handle terminal resize
                       if not self.error_handler.handle_resize(stdscr):
                           is_adequate, size_msg = self.error_handler.check_terminal_size(stdscr)
                           if not is_adequate:
                               stdscr.clear()
                               self.error_handler.safe_addstr(stdscr, 0, 0, "Terminal too small - please resize", curses.A_BOLD)
                               stdscr.refresh()
                               time.sleep(0.5)
                               continue
                       
                       # Clear screen and draw UI
                       stdscr.clear()
                       
                       # Draw all UI components with individual error handling
                       try:
                           self.draw_header(stdscr)
                       except Exception as e:
                           logging.error(f"Header draw failed: {e}")
                       
                       try:
                           self.draw_screen_content(stdscr)
                       except Exception as e:
                           logging.error(f"Content draw failed: {e}")
                       
                       try:
                           self.draw_input_bar(stdscr)
                       except Exception as e:
                           logging.error(f"Input bar draw failed: {e}")
                       
                       try:
                           self.draw_status_bar(stdscr)
                       except Exception as e:
                           logging.error(f"Status bar draw failed: {e}")
                       
                       # Refresh screen
                       try:
                           stdscr.refresh()
                       except curses.error:
                           # Screen refresh failed, try to continue
                           pass
                       
                       # Handle input with timeout for responsiveness
                       try:
                           stdscr.timeout(100)  # 100ms timeout
                           key = stdscr.getch()
                           stdscr.timeout(-1)  # Reset to blocking
                           
                           if key != -1:  # Key was pressed
                               self.handle_input_safely(key)
                           
                           # Reset error counter on successful loop
                           consecutive_errors = 0
                           
                       except curses.error:
                           # Input handling failed
                           consecutive_errors += 1
                           if consecutive_errors > max_consecutive_errors:
                               logging.error("Too many consecutive errors, exiting")
                               break
                           time.sleep(0.1)
                           continue
                       
                   except KeyboardInterrupt:
                       # Handle Ctrl+C gracefully
                       try:
                           self.safe_save_persistence()
                       except Exception:
                           pass
                       break
                   
                   except Exception as e:
                       # Catch-all error handler
                       logging.error(f"Main loop error: {e}", exc_info=True)
                       consecutive_errors += 1
                       self.session_stats['errors_encountered'] += 1
                       
                       if consecutive_errors > max_consecutive_errors:
                           logging.error("Too many consecutive errors, exiting")
                           break
                       
                       # Try to show error to user
                       try:
                           self.status_message = f"⚠️ System error: {str(e)[:50]}"
                       except Exception:
                           pass
                       
                       time.sleep(0.1)  # Brief pause before retry
           
           except Exception as e:
               # Critical error in main loop setup
               logging.critical(f"Critical error in main loop: {e}", exc_info=True)
               try:
                   stdscr.clear()
                   error_msg = f"Critical error: {str(e)}"
                   stdscr.addstr(0, 0, error_msg)
                   stdscr.addstr(1, 0, "Check coding_assistant_errors.log for details")
                   stdscr.addstr(2, 0, "Press any key to exit")
                   stdscr.refresh()
                   stdscr.getch()
               except Exception:
                   pass
       
       # Run with curses wrapper for proper cleanup
       try:
           curses.wrapper(main_loop)
       except Exception as e:
           # Final fallback error handling
           print(f"\n⚠️ Application failed to start: {e}")
           print("Check coding_assistant_errors.log for details")
           logging.critical(f"Application startup failed: {e}", exc_info=True)
           sys.exit(1)
       finally:
           # Final cleanup
           try:
               self.safe_save_persistence()
           except Exception as e:
               logging.error(f"Final save failed: {e}")

def main():
   """Main entry point with error handling"""
   try:
       # Check Python version
       if sys.version_info < (3, 7):
           print("Error: Python 3.7 or higher required")
           sys.exit(1)
       
       # Check required environment variables
       if not os.getenv("GROQ_API_KEY"):
           print("Error: GROQ_API_KEY environment variable is required")
           print("Please set your Groq API key: export GROQ_API_KEY=your_key_here")
           sys.exit(1)
       
       # Check if terminal supports curses
       try:
           import curses
           curses.setupterm()
       except Exception as e:
           print(f"Error: Terminal does not support curses: {e}")
           sys.exit(1)
       
       # Create and run application
       app = CodingAssistantCLI()
       app.run_safely()
       
   except KeyboardInterrupt:
       print("\nGoodbye!")
       sys.exit(0)
   except Exception as e:
       print(f"Startup error: {e}")
       logging.critical(f"Startup error: {e}", exc_info=True)
       sys.exit(1)

if __name__ == "__main__":
   main()
