/**
 * ツールレジストリ - Claude Code風のツールをOpenAI function calling形式で定義
 */
import type { ToolDefinition, ToolCall } from "../api/openai.js";
import { executeBash } from "./bash.js";
import { executeRead } from "./read.js";
import { executeWrite } from "./write.js";
import { executeEdit } from "./edit.js";
import { executeGlob } from "./glob.js";
import { executeGrep } from "./grep.js";

export interface ToolResult {
  output: string;
  error?: boolean;
}

export const TOOL_LABELS: Record<string, string> = {
  Bash: "Running",
  Read: "Reading",
  Write: "Writing",
  Edit: "Editing",
  Glob: "Searching",
  Grep: "Searching",
};

export const toolDefinitions: ToolDefinition[] = [
  {
    type: "function",
    function: {
      name: "Bash",
      description:
        "Execute a bash command and return its output. Use for system commands, git operations, running tests, installing packages, etc.",
      parameters: {
        type: "object",
        properties: {
          command: {
            type: "string",
            description: "The bash command to execute",
          },
          timeout: {
            type: "number",
            description: "Optional timeout in milliseconds (default 120000)",
          },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Read",
      description:
        "Read a file from the filesystem. Returns the file contents with line numbers.",
      parameters: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file to read",
          },
          offset: {
            type: "number",
            description: "Line number to start reading from (1-based)",
          },
          limit: {
            type: "number",
            description: "Number of lines to read",
          },
        },
        required: ["file_path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Write",
      description:
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
      parameters: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file to write",
          },
          content: {
            type: "string",
            description: "The content to write to the file",
          },
        },
        required: ["file_path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Edit",
      description:
        "Perform exact string replacement in a file. The old_string must be unique in the file.",
      parameters: {
        type: "object",
        properties: {
          file_path: {
            type: "string",
            description: "Absolute path to the file to edit",
          },
          old_string: {
            type: "string",
            description: "The exact string to find and replace",
          },
          new_string: {
            type: "string",
            description: "The replacement string",
          },
        },
        required: ["file_path", "old_string", "new_string"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Glob",
      description:
        "Find files matching a glob pattern. Returns matching file paths.",
      parameters: {
        type: "object",
        properties: {
          pattern: {
            type: "string",
            description: 'Glob pattern to match (e.g. "**/*.ts", "src/**/*.tsx")',
          },
          path: {
            type: "string",
            description: "Directory to search in (defaults to cwd)",
          },
        },
        required: ["pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "Grep",
      description:
        "Search file contents using regex. Returns matching lines with file paths and line numbers.",
      parameters: {
        type: "object",
        properties: {
          pattern: {
            type: "string",
            description: "Regex pattern to search for",
          },
          path: {
            type: "string",
            description: "File or directory to search in (defaults to cwd)",
          },
          include: {
            type: "string",
            description: 'File glob filter (e.g. "*.ts", "*.{js,jsx}")',
          },
        },
        required: ["pattern"],
      },
    },
  },
];

/**
 * ツールコールを実行して結果を返す
 */
export async function executeTool(toolCall: ToolCall): Promise<ToolResult> {
  const args = JSON.parse(toolCall.function.arguments);
  const name = toolCall.function.name;

  try {
    switch (name) {
      case "Bash":
        return await executeBash(args.command, args.timeout);
      case "Read":
        return await executeRead(args.file_path, args.offset, args.limit);
      case "Write":
        return await executeWrite(args.file_path, args.content);
      case "Edit":
        return await executeEdit(args.file_path, args.old_string, args.new_string);
      case "Glob":
        return await executeGlob(args.pattern, args.path);
      case "Grep":
        return await executeGrep(args.pattern, args.path, args.include);
      default:
        return { output: `Unknown tool: ${name}`, error: true };
    }
  } catch (err: any) {
    return { output: err.message || String(err), error: true };
  }
}
