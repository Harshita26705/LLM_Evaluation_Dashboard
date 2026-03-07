"""
Enhanced Code Analyzer using HuggingFace Transformers (No Ollama needed!)
Inspired by Google Gemini codebase analyzer
"""

import os
import ast
import re
import shutil
import requests
import json
from difflib import get_close_matches
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
                print("   🔧 Checking local AI model cache...")
                # Avoid network downloads/hangs: use local cache only.
                self.ai_pipeline = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device=-1,
                    max_length=512,
                    model_kwargs={"local_files_only": True}
                )
                self.use_ai = True
                print("   ✅ Local AI model loaded from cache")
            except Exception:
                self.ai_pipeline = None
                self.use_ai = False
                print("   ℹ️  Local AI model not cached. Using deterministic + Ollama fallback.")
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
                
            response = requests.post(url, json=payload, timeout=20)
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

        ai_improved_code = self._call_ollama(prompt, system_prompt)
        ai_improved_code = self._extract_code_block(ai_improved_code, language)
        ai_improved_code = self._left_align_code(ai_improved_code)

        if language == "python" and ai_improved_code and not self._is_valid_python(ai_improved_code):
            ai_improved_code = ""

        heuristic_improved_code = self._heuristic_improve_code(code, language)

        if self._is_meaningful_improvement(code, ai_improved_code, language):
            improved_code = ai_improved_code
            source = "ai"
        elif self._is_meaningful_improvement(code, heuristic_improved_code, language):
            improved_code = heuristic_improved_code
            source = "heuristic"
        else:
            improved_code = heuristic_improved_code if heuristic_improved_code else code
            source = "fallback"

        if language == "python" and improved_code and not self._is_valid_python(improved_code):
            improved_code = code
            source = "original"

        if not improved_code.strip():
            improved_code = code
            source = "original"

        return {
            "original_code": code,
            "improved_code": improved_code,
            "language": language,
            "improvement_source": source
        }
    
    def find_bugs(self, code: str, language: str = "python") -> Dict:
        """Find bugs in code (syntax + logic heuristics with deterministic summaries)."""

        bugs = []
        static_analysis = self._static_analysis(code, language)

        for err in static_analysis.get("errors", []):
            bugs.append({
                "severity": "Critical",
                "description": err,
                "suggestion": "Fix syntax errors before running code."
            })

        bugs.extend(self._logic_issue_heuristics(code, language))
        bugs = self._dedupe_bug_items(bugs)
        summary = self._build_bug_summary(bugs)

        return {
            "code": code,
            "language": language,
            "ai_analysis": {
                "bugs": bugs,
                "summary": summary
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
        """Generate project-level README documentation and save it to disk."""

        static_analysis = self._static_analysis(code, language)
        base_doc = self._basic_documentation(code, language, static_analysis)

        project_files = self._collect_project_files(os.getcwd())
        project_snapshot = "\n".join(f"- {path}" for path in project_files[:40])

        ai_doc = self._call_ollama(
            (
                f"Write a README for this project in Markdown.\n\n"
                f"Project files:\n{project_snapshot}\n\n"
                f"Key code snippet ({language}):\n{code}\n\n"
                "Include: overview, architecture, setup, run instructions, and key modules."
            ),
            "You are a technical writer. Use clear Markdown with practical developer guidance."
        )

        snippet_doc = ai_doc if ai_doc and not ai_doc.startswith("ERROR") else base_doc
        documentation = self._compose_project_readme(snippet_doc, project_files)

        readme_path = os.path.join(os.getcwd(), "README_GENERATED.md")
        with open(readme_path, "w", encoding="utf-8") as readme_file:
            readme_file.write(documentation)

        primary_readme_path = os.path.join(os.getcwd(), "README.md")
        primary_readme_created = False
        if not os.path.exists(primary_readme_path):
            with open(primary_readme_path, "w", encoding="utf-8") as primary_readme_file:
                primary_readme_file.write(documentation)
            primary_readme_created = True

        return {
            "code": code,
            "documentation": documentation,
            "format": "markdown",
            "readme_path": readme_path,
            "primary_readme_path": primary_readme_path,
            "primary_readme_created": primary_readme_created,
            "project_files_analyzed": len(project_files)
        }
    
    def analyze_llm_generated_code(self, original_prompt: str, generated_code: str, language: str = "python") -> Dict:
        """Evaluate if generated code matches prompt intent, runtime safety, and quality."""

        static_analysis = self._static_analysis(generated_code, language)
        base_quality = self._calculate_quality_score(static_analysis)

        prompt_keywords = self._extract_keywords(original_prompt)
        keyword_ratio = self._keyword_match_ratio(prompt_keywords, generated_code)

        prompt_requirements = self._prompt_requirement_score(original_prompt, generated_code, language)
        prompt_match = max(prompt_requirements["score"], (0.35 * keyword_ratio) + (0.65 * prompt_requirements["score"]))

        logic_issues = self._logic_issue_heuristics(generated_code, language)
        critical_logic_issues = [
            issue for issue in logic_issues
            if issue.get("severity", "").lower() in {"critical", "high"}
        ]

        issue_penalty = min(0.45, (0.12 * len(logic_issues)) + (0.20 * len(critical_logic_issues)))
        quality_score = max(0.0, min(1.0, (0.60 * base_quality) + (0.40 * prompt_match) - issue_penalty))

        matches_prompt = (
            prompt_match >= 0.65 and
            not static_analysis.get("errors") and
            not critical_logic_issues
        )

        ai_eval = self._call_ollama(
            f"Does this {language} code satisfy the prompt?\nPrompt: {original_prompt}\nCode:\n{generated_code}",
            "Answer briefly with issues and improvements."
        )

        analysis_parts = []
        if prompt_requirements["issues"]:
            analysis_parts.append("Prompt intent checks: " + "; ".join(prompt_requirements["issues"]))
        if logic_issues:
            analysis_parts.append("Runtime/logic issues: " + "; ".join(issue["description"] for issue in logic_issues))
        if ai_eval and not ai_eval.startswith("ERROR"):
            analysis_parts.append("AI review: " + ai_eval)

        combined_analysis = "\n".join(analysis_parts).strip()

        return {
            "original_prompt": original_prompt,
            "generated_code": generated_code,
            "language": language,
            "ai_analysis": {
                "evaluation": {
                    "matches_prompt": matches_prompt,
                    "match_ratio": round(prompt_match, 2),
                    "keyword_ratio": round(keyword_ratio, 2),
                    "prompt_requirement_score": round(prompt_requirements["score"], 2),
                    "quality_score": int(round(quality_score * 100)),
                    "issues": prompt_requirements["issues"] + [issue["description"] for issue in logic_issues],
                    "analysis": combined_analysis or ai_eval
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

        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        comment_lines = [line for line in lines if line.strip().startswith('#')]

        if language != "python":
            results["metrics"] = {
                "language": language,
                "Total Lines": len(lines),
                "Code Lines": len(non_empty_lines),
                "Comment Lines": len(comment_lines)
            }
            return results

        tree = None
        try:
            tree = ast.parse(code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            line_info = f"line {e.lineno}" if e.lineno else "unknown line"
            msg = e.msg if getattr(e, "msg", None) else str(e)
            results["errors"].append(f"Syntax Error ({line_info}): {msg}")

        functions = []
        classes = []
        imports = []
        if tree is not None:
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

        if language == "python":
            results["errors"].extend(self._python_syntax_heuristics(code))
            results["errors"].extend(self._undefined_return_heuristics(code))

        results["errors"] = self._dedupe_text_list(results["errors"])

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

        if not results["syntax_valid"]:
            results["suggestions"].append("⚠️  Syntax errors found. Fix the listed lines and re-run analysis to uncover deeper issues.")

        results["errors"] = self._dedupe_text_list(results["errors"])
        results["suggestions"] = self._dedupe_text_list(results["suggestions"])

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

    def _dedupe_text_list(self, values: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _python_syntax_heuristics(self, code: str) -> List[str]:
        issues = []
        lines = code.splitlines()

        block_prefixes = ("def ", "class ", "if ", "elif ", "for ", "while ", "with ")

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped.startswith(block_prefixes) and not stripped.endswith(":"):
                issues.append(f"Possible Syntax Error (line {line_number}): Missing ':' at end of statement.")

            if stripped == "else" or stripped == "finally":
                issues.append(f"Possible Syntax Error (line {line_number}): Missing ':' after '{stripped}'.")

            if stripped == "try":
                issues.append(f"Possible Syntax Error (line {line_number}): Missing ':' after 'try'.")

            if stripped.startswith("except") and not stripped.endswith(":"):
                issues.append(f"Possible Syntax Error (line {line_number}): Missing ':' after except block.")

            if re.search(r"print\(\s*(['\"][^'\"]*['\"])\s+[A-Za-z_][A-Za-z0-9_]*\s*\)", stripped):
                issues.append(f"Possible Syntax Error (line {line_number}): Missing comma or '+' in print arguments.")

        if code.count("(") != code.count(")"):
            issues.append("Possible Syntax Error: Unbalanced parentheses detected.")

        return issues

    def _undefined_return_heuristics(self, code: str) -> List[str]:
        assigned_variables = set()
        issues = []

        for line in code.splitlines():
            assignment = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if assignment:
                assigned_variables.add(assignment.group(1))

        for line_number, line in enumerate(code.splitlines(), start=1):
            return_match = re.match(r"^\s*return\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
            if not return_match:
                continue

            variable_name = return_match.group(1)
            if variable_name in {"True", "False", "None"} or variable_name in assigned_variables:
                continue

            close_match = get_close_matches(variable_name, list(assigned_variables), n=1, cutoff=0.75)
            if close_match:
                issues.append(
                    f"Possible Name Error (line {line_number}): `{variable_name}` may be a typo of `{close_match[0]}`."
                )
            else:
                issues.append(
                    f"Possible Name Error (line {line_number}): `{variable_name}` may be undefined before return."
                )

        return issues

    def _heuristic_improve_code(self, code: str, language: str) -> str:
        if not code.strip():
            return code

        if language == "python":
            return self._heuristic_improve_python(code)

        return self._left_align_code(code)

    def _heuristic_improve_python(self, code: str) -> str:
        if not self._is_valid_python(code):
            repaired_code = self._repair_invalid_python(code)
            if repaired_code and self._is_valid_python(repaired_code):
                code = repaired_code
            else:
                return self._left_align_code(repaired_code if repaired_code else code)

        try:
            tree = ast.parse(code)

            class PythonImprover(ast.NodeTransformer):
                def _ensure_docstring(self, node):
                    has_docstring = bool(ast.get_docstring(node))
                    if has_docstring:
                        return
                    doc = ast.Expr(value=ast.Constant(value=f"Auto-generated documentation for `{node.name}`."))
                    node.body.insert(0, doc)

                def visit_FunctionDef(self, node):
                    self.generic_visit(node)
                    self._ensure_docstring(node)
                    return node

                def visit_AsyncFunctionDef(self, node):
                    self.generic_visit(node)
                    self._ensure_docstring(node)
                    return node

                def visit_BinOp(self, node):
                    self.generic_visit(node)
                    if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                        if isinstance(node.right, ast.Constant) and node.right.value == 0:
                            node.right = ast.Constant(value=1)
                    return node

                def visit_ExceptHandler(self, node):
                    self.generic_visit(node)
                    if node.type is None:
                        node.type = ast.Name(id="Exception", ctx=ast.Load())
                    return node

            improved_tree = PythonImprover().visit(tree)
            ast.fix_missing_locations(improved_tree)

            if hasattr(ast, "unparse"):
                improved_code = ast.unparse(improved_tree)
                improved_code = self._left_align_code(improved_code)
                if improved_code and not improved_code.endswith("\n"):
                    improved_code += "\n"
                return improved_code
        except Exception:
            pass

        return self._left_align_code(code)

    def _repair_invalid_python(self, code: str) -> str:
        lines = code.splitlines()
        repaired_lines = []

        for line in lines:
            fixed = line.rstrip("\n")
            stripped = fixed.strip()

            if stripped:
                needs_colon = (
                    (stripped.startswith(("def ", "class ", "if ", "elif ", "for ", "while ", "with ")) and not stripped.endswith(":")) or
                    (stripped == "try") or
                    (stripped.startswith("except") and not stripped.endswith(":")) or
                    (stripped in {"else", "finally"})
                )
                if needs_colon:
                    fixed += ":"

                fixed = re.sub(
                    r"print\(\s*(['\"][^'\"]*['\"])\s+([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                    r"print(\1, \2)",
                    fixed,
                )

            repaired_lines.append(fixed)

        assigned_variables = []
        for line in repaired_lines:
            assignment = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if assignment:
                assigned_variables.append(assignment.group(1))

        normalized_lines = []
        for line in repaired_lines:
            return_match = re.match(r"^(\s*)return\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
            if return_match:
                indent, variable_name = return_match.groups()
                if variable_name not in assigned_variables and assigned_variables:
                    close_match = get_close_matches(variable_name, assigned_variables, n=1, cutoff=0.75)
                    if close_match:
                        line = f"{indent}return {close_match[0]}"
            normalized_lines.append(line)

        repaired_code = "\n".join(normalized_lines)
        if repaired_code and not repaired_code.endswith("\n"):
            repaired_code += "\n"
        return self._left_align_code(repaired_code)

    def _normalize_python_without_comments(self, code: str) -> str:
        normalized_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0]
            normalized_lines.append(line.strip())
        return "\n".join(normalized_lines)

    def _is_meaningful_improvement(self, original: str, candidate: str, language: str) -> bool:
        if not candidate or not candidate.strip():
            return False
        if original.strip() == candidate.strip():
            return False

        if language == "python":
            if not self._is_valid_python(candidate):
                return False

            original_norm = self._normalize_python_without_comments(original)
            candidate_norm = self._normalize_python_without_comments(candidate)
            if original_norm == candidate_norm:
                return False

        original_analysis = self._static_analysis(original, language)
        candidate_analysis = self._static_analysis(candidate, language)

        original_score = self._calculate_quality_score(original_analysis)
        candidate_score = self._calculate_quality_score(candidate_analysis)

        original_issues = len(self._logic_issue_heuristics(original, language))
        candidate_issues = len(self._logic_issue_heuristics(candidate, language))

        return (
            candidate_score >= original_score + 0.01 or
            candidate_issues < original_issues or
            original.strip() != candidate.strip()
        )

    def _extract_expected_print_value(self, prompt: str):
        lowered = prompt.lower()

        number_match = re.search(r"(?:print|output|display|show)\s+(-?\d+(?:\.\d+)?)", lowered)
        if number_match:
            value = number_match.group(1)
            return float(value) if "." in value else int(value)

        string_match = re.search(r"(?:print|output|display|show)\s+[\"']([^\"']+)[\"']", prompt)
        if string_match:
            return string_match.group(1)

        return None

    def _extract_python_print_values(self, code: str) -> Tuple[List, bool]:
        values = []
        has_print = False

        try:
            tree = ast.parse(code)
        except Exception:
            return values, has_print

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                has_print = True
                if not node.args:
                    continue
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    values.append(arg.value)
        return values, has_print

    def _has_division_by_zero(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except Exception:
            return "/0" in code.replace(" ", "")

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv)):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    return True
        return False

    def _prompt_requirement_score(self, prompt: str, code: str, language: str = "python") -> Dict:
        issues = []
        checks = {}
        score = 0.5

        if language != "python":
            keyword_ratio = self._keyword_match_ratio(self._extract_keywords(prompt), code)
            return {
                "score": round(min(1.0, 0.4 + (0.6 * keyword_ratio)), 3),
                "issues": issues,
                "checks": checks
            }

        try:
            ast.parse(code)
        except Exception:
            issues.append("Generated code has syntax errors.")
            return {"score": 0.0, "issues": issues, "checks": checks}

        expected_output = self._extract_expected_print_value(prompt)
        print_values, has_print = self._extract_python_print_values(code)
        checks["has_print"] = has_print

        if expected_output is not None:
            checks["expected_output"] = expected_output
            if not has_print:
                issues.append("Prompt asks for output/print, but no print statement was found.")
                score -= 0.35
            elif any(str(value) == str(expected_output) for value in print_values):
                score += 0.35
            else:
                readable_values = ", ".join(str(value) for value in print_values) if print_values else "non-literal expression"
                issues.append(f"Expected to print {expected_output}, but found {readable_values}.")
                score -= 0.35

        if self._has_division_by_zero(code):
            issues.append("Code contains division by zero.")
            score -= 0.45
            checks["division_by_zero"] = True

        if "function" in prompt.lower() or "def " in prompt.lower():
            tree = ast.parse(code)
            if not any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
                issues.append("Prompt implies a function, but no function definition was found.")
                score -= 0.2

        score = max(0.0, min(1.0, score))
        if not issues:
            score = max(score, 0.9)

        return {"score": round(score, 3), "issues": issues, "checks": checks}

    def _collect_project_files(self, project_root: str, max_files: int = 150) -> List[str]:
        ignored_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", "models"}
        tracked_extensions = {
            ".py", ".js", ".ts", ".tsx", ".html", ".css", ".md", ".json", ".bat", ".txt"
        }

        collected = []
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [directory for directory in dirs if directory not in ignored_dirs]

            for file_name in files:
                if len(collected) >= max_files:
                    return sorted(collected)

                extension = os.path.splitext(file_name)[1].lower()
                if extension not in tracked_extensions and not file_name.lower().startswith("readme"):
                    continue

                abs_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(abs_path, project_root).replace("\\", "/")
                collected.append(rel_path)

        return sorted(collected)

    def _compose_project_readme(self, generated_doc: str, project_files: List[str]) -> str:
        run_instructions = []
        if any(path.endswith("requirements.txt") for path in project_files):
            run_instructions.append("pip install -r requirements.txt")
        if any(path.endswith("requirements_flask.txt") for path in project_files):
            run_instructions.append("pip install -r requirements_flask.txt")
        if any(path.endswith("flask_app.py") for path in project_files):
            run_instructions.append("python flask_app.py")
        elif any(path.endswith("app.py") for path in project_files):
            run_instructions.append("python app.py")

        lines = [
            "# Project README",
            "",
            "## Overview",
            "This README is auto-generated from the current project structure and provided code context.",
            "",
            "## Project Structure (Key Files)",
        ]

        for path in project_files[:60]:
            lines.append(f"- `{path}`")

        lines.extend([
            "",
            "## Setup & Run",
        ])

        if run_instructions:
            for command in run_instructions:
                lines.append(f"- `{command}`")
        else:
            lines.append("- Add your setup/run steps here.")

        lines.extend([
            "",
            "## Detailed Documentation",
            generated_doc.strip() or "No additional documentation available.",
            ""
        ])

        return "\n".join(lines)

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

    def _dedupe_bug_items(self, bugs: List[Dict]) -> List[Dict]:
        deduped = []
        seen = set()
        for bug in bugs:
            severity = str(bug.get("severity", "")).strip().lower()
            description = str(bug.get("description", "")).strip().lower()
            key = (severity, description)
            if not description or key in seen:
                continue
            seen.add(key)
            deduped.append(bug)
        return deduped

    def _build_bug_summary(self, bugs: List[Dict]) -> str:
        if not bugs:
            return "No obvious bugs detected in this snippet."

        highlights = []
        for bug in bugs[:3]:
            severity = bug.get("severity", "Issue")
            description = bug.get("description", "").strip()
            if description:
                highlights.append(f"{severity}: {description}")

        base = f"Detected {len(bugs)} potential issue(s)."
        if highlights:
            return base + " " + " | ".join(highlights)
        return base

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
