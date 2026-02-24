"""
Enhanced Code Analyzer using HuggingFace Transformers (No Ollama needed!)
Inspired by Google Gemini codebase analyzer
"""

import os
import ast
import shutil
import requests
import json
from typing import Dict, List, Optional, Tuple
from collections import Counter

# Try to import optional dependencies
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

try:
    from gitingest import ingest
    GITINGEST_AVAILABLE = True
except ImportError:
    GITINGEST_AVAILABLE = False

# Import transformers for local AI
try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("   ✅ Transformers available for local AI")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("   ⚠️  Transformers not available")


class OllamaCodeAnalyzer:
    """Code analyzer using HuggingFace transformers (no Ollama needed!)"""
    
    def __init__(self, model="llama3.2", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.repo_dir = "./temp_repo"
        
        # Initialize local AI model
        self.ai_pipeline = None
        self.use_ai = False
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print("   🔧 Loading local AI model (first time may take 1-2 min)...")
                # Use a small, fast model for code analysis
                self.ai_pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",  # Small 350MB model
                    device=-1,  # CPU
                    max_length=512
                )
                self.use_ai = True
                print("   ✅ Local AI model loaded - AI features ready!")
            except Exception as e:
                print(f"   ⚠️  Could not load AI model: {e}")
                self.ai_pipeline = None
        else:
            print("   ℹ️  Install transformers for AI features: pip install transformers")
        
    def _call_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Generate AI response using local model or Ollama"""
        
        # Try local AI first
        if self.use_ai and self.ai_pipeline:
            try:
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                # Truncate prompt to avoid issues
                if len(full_prompt) > 800:
                    full_prompt = full_prompt[:800] + "..."
                
                response = self.ai_pipeline(
                    full_prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.ai_pipeline.tokenizer.eos_token_id
                )
                
                generated = response[0]['generated_text']
                # Remove the prompt from response
                result = generated[len(full_prompt):].strip()
                
                return result if result else "Analysis complete"
                
            except Exception as e:
                print(f"   ⚠️ AI generation failed: {e}")
                return f"Basic analysis available (AI error: {str(e)[:50]})"
        
        # Fallback to Ollama (if user has it)
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
                
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.ConnectionError:
            return "ERROR: No AI model available. Install local model or start Ollama."
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def analyze_code_snippet(self, code: str, language: str = "python") -> Dict:
        """Full analysis: score, suggestions, and improved code"""

        static_analysis = self._static_analysis(code, language)
        quality_score = self._calculate_quality_score(static_analysis)

        improved = self.generate_improved_code(code, language)

        return {
            **static_analysis,
            "quality_score": quality_score,
            "ai_analysis": {
                "score": int(round(quality_score * 100)),
                "suggestions": static_analysis.get("suggestions", []),
                "improved_code": improved.get("improved_code", code)
            },
            "analysis_type": "full"
        }
    
    def generate_improved_code(self, code: str, language: str = "python") -> Dict:
        """Generate improved version of code"""

        system_prompt = "You are an expert programmer. Improve code while maintaining functionality. Return only code."

        prompt = f"""Improve this {language} code while preserving its behavior.

Original code:
```{language}
{code}
```

Return ONLY the improved code. Keep valid syntax and left-aligned formatting."""

        improved_code = self._call_ollama(prompt, system_prompt)
        improved_code = self._extract_code_block(improved_code, language)
        improved_code = self._left_align_code(improved_code)

        if language == "python" and not self._is_valid_python(improved_code):
            improved_code = code

        return {
            "original_code": code,
            "improved_code": improved_code,
            "language": language
        }
    
    def find_bugs(self, code: str, language: str = "python") -> Dict:
        """Find bugs in code (syntax + logic heuristics + AI hints)"""

        bugs = []
        static_analysis = self._static_analysis(code, language)

        for err in static_analysis.get("errors", []):
            bugs.append({
                "severity": "Critical",
                "description": err,
                "suggestion": "Fix syntax errors before running code."
            })

        bugs.extend(self._logic_issue_heuristics(code, language))

        ai_hint = self._call_ollama(
            f"List potential bugs in this {language} code:\n\n{code}",
            "You are a code reviewer. Keep answers short."
        )

        return {
            "code": code,
            "language": language,
            "ai_analysis": {
                "bugs": bugs,
                "summary": ai_hint
            }
        }
    
    def security_analysis(self, code: str, language: str = "python") -> Dict:
        """Check for hacking/data leak/safety risks"""

        security_issues = self._security_issue_heuristics(code, language)

        ai_hint = self._call_ollama(
            f"Identify security risks in this {language} code:\n\n{code}",
            "You are a security reviewer. Keep answers short."
        )

        return {
            "language": language,
            "ai_analysis": {
                "security": security_issues,
                "summary": ai_hint
            }
        }
    
    def generate_documentation(self, code: str, language: str = "python") -> Dict:
        """Generate documentation for code"""

        static_analysis = self._static_analysis(code, language)
        base_doc = self._basic_documentation(code, language, static_analysis)

        ai_doc = self._call_ollama(
            f"Write brief documentation for this {language} code:\n\n{code}",
            "You are a technical writer. Use Markdown."
        )

        documentation = ai_doc if ai_doc and not ai_doc.startswith("ERROR") else base_doc

        return {
            "code": code,
            "documentation": documentation,
            "format": "markdown"
        }
    
    def analyze_llm_generated_code(self, original_prompt: str, generated_code: str, language: str = "python") -> Dict:
        """Evaluate if generated code matches the prompt and its quality"""

        static_analysis = self._static_analysis(generated_code, language)
        quality_score = self._calculate_quality_score(static_analysis)

        prompt_keywords = self._extract_keywords(original_prompt)
        match_ratio = self._keyword_match_ratio(prompt_keywords, generated_code)
        matches_prompt = match_ratio >= 0.2

        ai_eval = self._call_ollama(
            f"Does this {language} code satisfy the prompt?\nPrompt: {original_prompt}\nCode:\n{generated_code}",
            "Answer briefly with issues and improvements."
        )

        return {
            "original_prompt": original_prompt,
            "generated_code": generated_code,
            "language": language,
            "ai_analysis": {
                "evaluation": {
                    "matches_prompt": matches_prompt,
                    "match_ratio": round(match_ratio, 2),
                    "quality_score": int(round(quality_score * 100)),
                    "analysis": ai_eval
                }
            },
            "quality_score": quality_score,
            "suggestions": static_analysis.get("suggestions", [])
        }
    
    def analyze_repository(self, repo_url: str) -> Dict:
        """Analyze entire GitHub repository"""
        
        if not GITINGEST_AVAILABLE:
            return {"error": "gitingest not installed. Run: pip install gitingest"}
        
        if not GIT_AVAILABLE:
            return {"error": "GitPython not installed. Run: pip install gitpython"}
        
        try:
            # Clone repository
            if os.path.exists(self.repo_dir):
                shutil.rmtree(self.repo_dir)
            
            git.Repo.clone_from(repo_url, self.repo_dir, depth=1)
            
            # Extract codebase content
            _, tree, content = ingest(self.repo_dir)
            
            # Analyze with AI
            system_prompt = "You are a software architect and code reviewer."
            
            prompt = f"""Analyze this entire codebase:

**Directory Structure:**
{tree[:2000]}

**Code Content:**
{content[:8000]}

Provide:
1. Project summary
2. Architecture overview
3. Technology stack
4. Top 3 strengths
5. Top 3 areas for improvement
6. Security concerns
7. Scalability assessment

Be concise but comprehensive."""

            analysis = self._call_ollama(prompt, system_prompt)
            
            # Clean up
            shutil.rmtree(self.repo_dir)
            
            return {
                "repo_url": repo_url,
                "directory_tree": tree[:1000],
                "analysis": analysis,
                "analysis_type": "repository"
            }
            
        except Exception as e:
            return {"error": f"Repository analysis failed: {str(e)}"}
    
    def analyze_git_diff(self, repo_path: str) -> Dict:
        """Analyze git diff for changelog"""
        
        if not GIT_AVAILABLE:
            return {"error": "GitPython not installed"}
        
        try:
            repo = git.Repo(repo_path)
            branch_name = "main"
            
            # Get commit IDs
            commit_ids = [commit.hexsha for commit in repo.iter_commits(branch_name)]
            
            if len(commit_ids) < 2:
                return {"error": "Not enough commits to compare"}
            
            # Get diff
            diff_text = repo.git.diff(commit_ids[0], commit_ids[1])
            
            # Analyze with AI
            system_prompt = "You are a git changelog expert."
            
            prompt = f"""Analyze this git diff and create a changelog:

```diff
{diff_text[:5000]}
```

Provide:
1. Summary of changes
2. Breaking changes (if any)
3. New features
4. Bug fixes
5. Performance improvements
6. Documentation updates

Use Markdown format."""

            changelog = self._call_ollama(prompt, system_prompt)
            
            return {
                "diff": diff_text[:1000],
                "changelog": changelog,
                "commits_compared": commit_ids[:2]
            }
            
        except Exception as e:
            return {"error": f"Git diff analysis failed: {str(e)}"}
    
    def _static_analysis(self, code: str, language: str) -> Dict:
        """Static analysis for metrics, errors, and suggestions"""

        results = {
            "syntax_valid": False,
            "errors": [],
            "metrics": {},
            "suggestions": []
        }

        if language != "python":
            results["metrics"] = {"language": language, "lines": len(code.split('\n'))}
            return results

        try:
            tree = ast.parse(code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["errors"].append(f"Syntax Error: {str(e)}")
            return results

        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        comment_lines = [l for l in lines if l.strip().startswith('#')]

        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]

        results["metrics"] = {
            "Total Lines": len(lines),
            "Code Lines": len(non_empty_lines),
            "Comment Lines": len(comment_lines),
            "Functions": len(functions),
            "Classes": len(classes),
            "Imports": len(imports)
        }

        if len(comment_lines) == 0:
            results["suggestions"].append("⚠️  No comments found. Add documentation.")
        if 'try:' not in code and 'except' not in code:
            results["suggestions"].append("⚠️  No error handling detected.")
        if len(non_empty_lines) > 120:
            results["suggestions"].append("⚠️  Code is long. Consider splitting functions.")
        if not functions and len(non_empty_lines) > 20:
            results["suggestions"].append("⚠️  No functions found. Consider modularizing.")
        if 'eval(' in code or 'exec(' in code:
            results["errors"].append("🚨 Security Issue: eval/exec usage detected.")

        return results

    def _calculate_quality_score(self, analysis: Dict) -> float:
        metrics = analysis.get("metrics", {})
        errors = analysis.get("errors", [])
        suggestions = analysis.get("suggestions", [])

        total_lines = metrics.get("Total Lines", 1) or 1
        comment_lines = metrics.get("Comment Lines", 0)
        code_lines = metrics.get("Code Lines", 1) or 1
        functions = metrics.get("Functions", 0)

        comment_ratio = comment_lines / total_lines
        avg_fn_len = code_lines / max(functions, 1)

        score = 0.9
        if errors:
            score -= 0.25
        score -= min(0.3, len(suggestions) * 0.04)

        if comment_ratio >= 0.15:
            score += 0.05
        elif comment_ratio < 0.03:
            score -= 0.12
        elif comment_ratio < 0.08:
            score -= 0.05

        if avg_fn_len > 120:
            score -= 0.15
        elif avg_fn_len > 60:
            score -= 0.07
        elif avg_fn_len < 20:
            score += 0.03

        if total_lines > 300:
            score -= 0.1
        elif total_lines < 15:
            score -= 0.05

        if functions == 0 and total_lines > 30:
            score -= 0.05

        return round(max(0.1, min(0.98, score)), 3)

    def _extract_code_block(self, text: str, language: str) -> str:
        if not text:
            return ""
        if "```" not in text:
            return text.strip()
        parts = text.split("```")
        for part in parts:
            if language in part.lower() or "python" in part.lower():
                return part.replace(language, "").replace("python", "").strip()
        return parts[-1].strip()

    def _left_align_code(self, code: str) -> str:
        lines = code.split("\n")
        stripped = [line.rstrip() for line in lines]
        return "\n".join(stripped).lstrip("\n")

    def _is_valid_python(self, code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    def _logic_issue_heuristics(self, code: str, language: str) -> List[Dict]:
        issues = []
        if language == "python":
            if "/ 0" in code or "/0" in code:
                issues.append({
                    "severity": "High",
                    "description": "Division by zero detected.",
                    "suggestion": "Check denominator before division."
                })
            if "input(" in code and "int(" not in code:
                issues.append({
                    "severity": "Medium",
                    "description": "User input used without type validation.",
                    "suggestion": "Validate and convert input safely."
                })
        return issues

    def _security_issue_heuristics(self, code: str, language: str) -> List[Dict]:
        issues = []
        if language == "python":
            if "eval(" in code or "exec(" in code:
                issues.append({"type": "Code Execution", "description": "eval/exec can run arbitrary code."})
            if "pickle.loads" in code:
                issues.append({"type": "Deserialization", "description": "pickle.loads can execute arbitrary code."})
            if "subprocess" in code and "shell=True" in code:
                issues.append({"type": "Command Injection", "description": "subprocess with shell=True is risky."})
            if "input(" in code and "sql" in code.lower():
                issues.append({"type": "SQL Injection", "description": "Possible SQL injection risk."})
            if "password" in code.lower() or "token" in code.lower() or "secret" in code.lower():
                if "print(" in code or "log" in code.lower():
                    issues.append({"type": "Data Leak", "description": "Sensitive data may be logged or printed."})
        return issues

    def _basic_documentation(self, code: str, language: str, analysis: Dict) -> str:
        metrics = analysis.get("metrics", {})
        functions = [node.name for node in ast.walk(ast.parse(code)) if isinstance(node, ast.FunctionDef)] if language == "python" and analysis.get("syntax_valid") else []
        classes = [node.name for node in ast.walk(ast.parse(code)) if isinstance(node, ast.ClassDef)] if language == "python" and analysis.get("syntax_valid") else []

        doc = ["# Code Documentation", "", "## Overview", "This module provides the following elements:", ""]
        if functions:
            doc.append("## Functions")
            for fn in functions:
                doc.append(f"- `{fn}()`")
            doc.append("")
        if classes:
            doc.append("## Classes")
            for cls in classes:
                doc.append(f"- `{cls}`")
            doc.append("")
        doc.append("## Metrics")
        for k, v in metrics.items():
            doc.append(f"- {k}: {v}")
        return "\n".join(doc)

    def _extract_keywords(self, prompt: str) -> List[str]:
        words = [w.strip().lower() for w in prompt.split()]
        return [w for w in words if len(w) > 3]

    def _keyword_match_ratio(self, keywords: List[str], code: str) -> float:
        if not keywords:
            return 0.0
        code_lower = code.lower()
        matches = sum(1 for w in keywords if w in code_lower)
        return matches / len(keywords)


# Singleton instance
_analyzer = None

def get_analyzer(model="llama3.2"):
    """Get or create analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = OllamaCodeAnalyzer(model=model)
    return _analyzer
