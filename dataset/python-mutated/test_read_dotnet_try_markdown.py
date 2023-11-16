import jupytext
from jupytext.cell_metadata import parse_key_equal_value
from jupytext.compare import compare

def test_parse_metadata():
    if False:
        while True:
            i = 10
    assert parse_key_equal_value('--key value --key-2 .\\a\\b.cs') == {'incorrectly_encoded_metadata': '--key value --key-2 .\\a\\b.cs'}

def test_parse_double_hyphen_metadata():
    if False:
        for i in range(10):
            print('nop')
    assert parse_key_equal_value('--key1 value1 --key2 value2') == {'incorrectly_encoded_metadata': '--key1 value1 --key2 value2'}

def test_read_dotnet_try_markdown(md='This is a dotnet/try Markdown file, inspired\nfrom this [post](https://devblogs.microsoft.com/dotnet/creating-interactive-net-documentation/)\n\n``` cs --region methods --source-file .\\myapp\\Program.cs --project .\\myapp\\myapp.csproj\nvar name ="Rain";\nConsole.WriteLine($"Hello {name.ToUpper()}!");\n```\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(md, fmt='.md')
    assert nb.metadata['jupytext']['main_language'] == 'csharp'
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'var name ="Rain";\nConsole.WriteLine($"Hello {name.ToUpper()}!");'
    compare(nb.cells[1].metadata, {'language': 'cs', 'incorrectly_encoded_metadata': '--region methods --source-file .\\myapp\\Program.cs --project .\\myapp\\myapp.csproj'})
    md2 = jupytext.writes(nb, 'md')
    compare(md2, md.replace('``` cs', '```cs'))