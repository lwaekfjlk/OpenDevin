from opendevin.runtime.plugins import SWEAgentCommandsRequirement

_SWEAGENT_BASH_DOCS = '\n'.join(
    filter(
        lambda x: not x.startswith('submit'),
        SWEAgentCommandsRequirement.documentation.split('\n'),
    )
)
# _SWEAGENT_BASH_DOCS content below:
"""
open <path> [<line_number>] - opens the file at the given path in the editor. If line_number is provided, the window will be move to include that line
goto <line_number> - moves the window to show <line_number>
scroll_down - moves the window down {WINDOW} lines
scroll_up - moves the window down {WINDOW} lines
create <filename> - creates and opens a new file with the given name
search_dir <search_term> [<dir>] - searches for search_term in all files in dir. If dir is not provided, searches in the current directory
search_file <search_term> [<file>] - searches for search_term in file. If file is not provided, searches in the current open file
find_file <file_name> [<dir>] - finds all files with the given name in dir. If dir is not provided, searches in the current directory
edit <start_line>:<end_line>
<replacement_text>
end_of_edit - replaces lines <start_line> through <end_line> (inclusive) with the given text in the open file. The replacement text is terminated by a line with only end_of_edit on it. All of the <replacement text> will be entered, so make sure your indentation is formatted properly. Python files will be checked for syntax errors after the edit. If the system detects a syntax error, the edit will not be executed. Simply try to edit the file again, but make sure to read the error message and modify the edit command you issue accordingly. Issuing the same command a second time will just lead to the same error message again. Remember, the file must be open before editing.
"""

COMMAND_DOCS = (
    '\nApart from the standard bash commands, you can also use the following special commands in <execute_bash> environment:\n'
    f'{_SWEAGENT_BASH_DOCS}'
    "Please note that THE EDIT COMMAND REQUIRES PROPER INDENTATION. If you'd like to add the line '        print(x)' you must fully write that out, with all those spaces before the code! Indentation is important and code that is not indented correctly will fail and require fixing before it can be run."
)

SYSTEM_PREFIX = ''

GITHUB_MESSAGE = """To do any activities on GitHub, you should use the token in the $GITHUB_TOKEN environment variable.
For instance, to push a local branch `my_branch` to the github repo `owner/repo`, you can use the following four commands:
<execute_bash> git push https://$GITHUB_TOKEN@github.com/owner/repo.git my_branch </execute_bash>
If you require access to GitHub but $GITHUB_TOKEN is not set, ask the user to set it for you."""

SYSTEM_SUFFIX = """The assistant's response should be concise.
You should include <execute_ipython> or <execute_bash> or <execute_browse> in every one of your responses, unless you are finished with the task or need more input or action from the user in order to proceed.
IMPORTANT: Whenever possible, execute the code for the user using <execute_ipython> or <execute_bash> or <execute_browse> instead of providing it.
"""

EXAMPLES = ''

INVALID_INPUT_MESSAGE = (
    "I don't understand your input. \n"
    'If you want to execute a bash command, please use <execute_bash> YOUR_COMMAND_HERE </execute_bash>.\n'
    'If you want to execute a block of Python code, please use <execute_ipython> YOUR_COMMAND_HERE </execute_ipython>.\n'
    'If you want to browse the Internet, please use <execute_browse> YOUR_COMMAND_HERE </execute_browse>.\n'
)
