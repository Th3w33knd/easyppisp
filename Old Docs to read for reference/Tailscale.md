### **The "Master Remote Environment" Prompt**

"I am working on a software project where the development environment is a remote **GitHub Codespace (Ubuntu 24.04)**, not my local machine. I access this environment via **Tailscale SSH**.

**Connection Details:**

* **SSH Target:** `codespace@100.94.84.29`
* **Project Root:** `/workspaces/easyppisp`
* **Primary Shell:** Bash

**Your Instructions:**

1. **Remote Execution Only:** All terminal commands, script executions, and testing (e.g., `pytest`, `python3`, `pip install`) must be formatted to be run inside the remote SSH session. Do not suggest local Windows commands.
2. **Path Awareness:** Always assume the working directory is `/workspaces/easyppisp`. When referencing files, use paths relative to this root.
3. **File Management:** If you need me to create or edit a file, provide the full content so I can use `cat <<EOF > filename` or `nano` within the SSH session. If a file needs to be moved from my local Windows machine to the host, provide the specific `scp` command.
4. **Environment Constraints:** This is a Codespace running Tailscale in **userspace-networking mode**. Be aware that certain low-level networking commands (like `ping` or raw socket manipulation) may behave differently, but standard web servers and application code will work fine via port forwarding.
5. **Contextual Intelligence:** Before suggesting a fix, check if it requires a specific Linux dependency. If so, include the `sudo apt-get install` command in your response.

Please acknowledge that you understand all operations must happen on the remote host at `100.94.84.29` and that you are ready to begin coding in the `easyppisp` project."

---

### **CRUD Operations for AI Agent (SSH Terminal Edition)**

#### **1. CREATE / OVERWRITE (The "Heredoc" Method)**

The agent should use this to create new files or completely overwrite existing ones.

```bash
cat << 'EOF' > /workspaces/easyppisp/src/example.py
def hello_world():
    print("Hello from Tailscale SSH!")

if __name__ == "__main__":
    hello_world()
EOF

```

*Note: Using `'EOF'` in quotes prevents the shell from trying to expand variables like `$HOME` inside the file.*

#### **2. READ (Scoping the File)**

To see what’s inside a file before editing:

```bash
cat /workspaces/easyppisp/src/example.py

```

#### **3. UPDATE (Appending or Patching)**

* **To Append to the end:** Change `>` to `>>`.
* **To Edit specific lines:** The agent should use `sed`. For example, to change "Hello" to "Greetings":

```bash
sed -i 's/Hello/Greetings/g' /workspaces/easyppisp/src/example.py

```

#### **4. DELETE**

```bash
rm /workspaces/easyppisp/src/example.py

```

---

### **The Updated "All-In-One" Master Prompt**

"I am working on a software project in a remote **GitHub Codespace (Ubuntu 24.04)** via **Tailscale SSH**.

**Connection:** `ssh codespace@100.94.84.29`
**Root:** `/workspaces/easyppisp`

**Operational Rules for File Management:**

1. **Writing Files:** Use `cat << 'EOF' > path/to/file` for all file creations or full overwrites. This ensures code integrity across the SSH tunnel.
2. **Directory Management:** Always check if a directory exists using `mkdir -p` before writing a file to a new path.
3. **Verification:** After writing a file, optionally run `ls -l` or `checksum` if the file is critical.
4. **Python Environment:** Use `python3` and `pip` as they are configured in this Ubuntu environment.
5. **Permissions:** Use `sudo` only when modifying system-level packages; stay in the `codespace` user context for all project file changes.

Please use these CRUD patterns to manage the codebase at `/workspaces/easyppisp`."

---