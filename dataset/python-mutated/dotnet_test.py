from __future__ import annotations
from pre_commit.languages import dotnet
from testing.language_helpers import run_language

def _write_program_cs(tmp_path):
    if False:
        return 10
    program_cs = 'using System;\n\nnamespace dotnet_tests\n{\n    class Program\n    {\n        static void Main(string[] args)\n        {\n            Console.WriteLine("Hello from dotnet!");\n        }\n    }\n}\n'
    tmp_path.joinpath('Program.cs').write_text(program_cs)

def _csproj(tool_name):
    if False:
        while True:
            i = 10
    return f'<Project Sdk="Microsoft.NET.Sdk">\n  <PropertyGroup>\n    <OutputType>Exe</OutputType>\n    <TargetFramework>net6</TargetFramework>\n    <PackAsTool>true</PackAsTool>\n    <ToolCommandName>{tool_name}</ToolCommandName>\n    <PackageOutputPath>./nupkg</PackageOutputPath>\n  </PropertyGroup>\n</Project>\n'

def test_dotnet_csproj(tmp_path):
    if False:
        print('Hello World!')
    csproj = _csproj('testeroni')
    _write_program_cs(tmp_path)
    tmp_path.joinpath('dotnet_csproj.csproj').write_text(csproj)
    ret = run_language(tmp_path, dotnet, 'testeroni')
    assert ret == (0, b'Hello from dotnet!\n')

def test_dotnet_csproj_prefix(tmp_path):
    if False:
        print('Hello World!')
    csproj = _csproj('testeroni.tool')
    _write_program_cs(tmp_path)
    tmp_path.joinpath('dotnet_hooks_csproj_prefix.csproj').write_text(csproj)
    ret = run_language(tmp_path, dotnet, 'testeroni.tool')
    assert ret == (0, b'Hello from dotnet!\n')

def test_dotnet_sln(tmp_path):
    if False:
        while True:
            i = 10
    csproj = _csproj('testeroni')
    sln = 'Microsoft Visual Studio Solution File, Format Version 12.00\n# Visual Studio 15\nVisualStudioVersion = 15.0.26124.0\nMinimumVisualStudioVersion = 15.0.26124.0\nProject("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "dotnet_hooks_sln_repo", "dotnet_hooks_sln_repo.csproj", "{6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}"\nEndProject\nGlobal\n        GlobalSection(SolutionConfigurationPlatforms) = preSolution\n                Debug|Any CPU = Debug|Any CPU\n                Debug|x64 = Debug|x64\n                Debug|x86 = Debug|x86\n                Release|Any CPU = Release|Any CPU\n                Release|x64 = Release|x64\n                Release|x86 = Release|x86\n        EndGlobalSection\n        GlobalSection(SolutionProperties) = preSolution\n                HideSolutionNode = FALSE\n        EndGlobalSection\n        GlobalSection(ProjectConfigurationPlatforms) = postSolution\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|Any CPU.ActiveCfg = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|Any CPU.Build.0 = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|x64.ActiveCfg = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|x64.Build.0 = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|x86.ActiveCfg = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Debug|x86.Build.0 = Debug|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|Any CPU.ActiveCfg = Release|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|Any CPU.Build.0 = Release|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|x64.ActiveCfg = Release|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|x64.Build.0 = Release|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|x86.ActiveCfg = Release|Any CPU\n                {6568CFDB-6F6F-45A9-932C-8C7DAABC8E56}.Release|x86.Build.0 = Release|Any CPU\n        EndGlobalSection\nEndGlobal\n'
    _write_program_cs(tmp_path)
    tmp_path.joinpath('dotnet_hooks_sln_repo.csproj').write_text(csproj)
    tmp_path.joinpath('dotnet_hooks_sln_repo.sln').write_text(sln)
    ret = run_language(tmp_path, dotnet, 'testeroni')
    assert ret == (0, b'Hello from dotnet!\n')

def _setup_dotnet_combo(tmp_path):
    if False:
        print('Hello World!')
    sln = 'Microsoft Visual Studio Solution File, Format Version 12.00\n# Visual Studio Version 16\nVisualStudioVersion = 16.0.30114.105\nMinimumVisualStudioVersion = 10.0.40219.1\nProject("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "proj1", "proj1\\proj1.csproj", "{38A939C3-DEA4-47D7-9B75-0418C4249662}"\nEndProject\nProject("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "proj2", "proj2\\proj2.csproj", "{4C9916CB-165C-4EF5-8A57-4CB6794C1EBF}"\nEndProject\nGlobal\n        GlobalSection(SolutionConfigurationPlatforms) = preSolution\n                Debug|Any CPU = Debug|Any CPU\n                Release|Any CPU = Release|Any CPU\n        EndGlobalSection\n        GlobalSection(SolutionProperties) = preSolution\n                HideSolutionNode = FALSE\n        EndGlobalSection\n        GlobalSection(ProjectConfigurationPlatforms) = postSolution\n                {38A939C3-DEA4-47D7-9B75-0418C4249662}.Debug|Any CPU.ActiveCfg = Debug|Any CPU\n                {38A939C3-DEA4-47D7-9B75-0418C4249662}.Debug|Any CPU.Build.0 = Debug|Any CPU\n                {38A939C3-DEA4-47D7-9B75-0418C4249662}.Release|Any CPU.ActiveCfg = Release|Any CPU\n                {38A939C3-DEA4-47D7-9B75-0418C4249662}.Release|Any CPU.Build.0 = Release|Any CPU\n                {4C9916CB-165C-4EF5-8A57-4CB6794C1EBF}.Debug|Any CPU.ActiveCfg = Debug|Any CPU\n                {4C9916CB-165C-4EF5-8A57-4CB6794C1EBF}.Debug|Any CPU.Build.0 = Debug|Any CPU\n                {4C9916CB-165C-4EF5-8A57-4CB6794C1EBF}.Release|Any CPU.ActiveCfg = Release|Any CPU\n                {4C9916CB-165C-4EF5-8A57-4CB6794C1EBF}.Release|Any CPU.Build.0 = Release|Any CPU\n        EndGlobalSection\nEndGlobal\n'
    tmp_path.joinpath('dotnet_hooks_combo_repo.sln').write_text(sln)
    csproj1 = _csproj('proj1')
    proj1 = tmp_path.joinpath('proj1')
    proj1.mkdir()
    proj1.joinpath('proj1.csproj').write_text(csproj1)
    _write_program_cs(proj1)
    csproj2 = _csproj('proj2')
    proj2 = tmp_path.joinpath('proj2')
    proj2.mkdir()
    proj2.joinpath('proj2.csproj').write_text(csproj2)
    _write_program_cs(proj2)

def test_dotnet_combo_proj1(tmp_path):
    if False:
        return 10
    _setup_dotnet_combo(tmp_path)
    ret = run_language(tmp_path, dotnet, 'proj1')
    assert ret == (0, b'Hello from dotnet!\n')

def test_dotnet_combo_proj2(tmp_path):
    if False:
        i = 10
        return i + 15
    _setup_dotnet_combo(tmp_path)
    ret = run_language(tmp_path, dotnet, 'proj2')
    assert ret == (0, b'Hello from dotnet!\n')