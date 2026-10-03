"""Minimal GTP client for driving leelaz from the helper scripts."""
import subprocess


class GTPEngine:
    def __init__(self, cmd, name=None):
        self.name = name or cmd[0]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def send(self, command):
        """Send one command; return the response text without the '=' marker."""
        self.p.stdin.write(command + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("%s exited while running: %s" % (self.name, command))
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line.strip())
        reply = "\n".join(lines)
        if reply.startswith("?"):
            raise RuntimeError("%s: '%s' failed: %s" % (self.name, command, reply))
        return reply[1:].strip()

    def close(self):
        try:
            self.send("quit")
        except Exception:
            pass
        self.p.wait(timeout=10)
