class Chunk:
    code: str = ""
    embeddings: list = []
    filename: str = "unknown"
    start_index: int = -1
    end_index: int = -1

    def __init__(
        self,
        code: str = "",
        embeddings: list = [],
        filename: str = "unknown",
        start_index: int = -1,
        end_index: int = -1,
    ):
        self.code = code
        self.embeddings = embeddings
        self.filename = filename
        self.start_index = start_index
        self.end_index = end_index

    def __str__(self):
        return f"Filename: {self.filename} | Starting_line_number: {self.start_index} | Ending_line_number: {self.end_index} | Code: '{self.code}'"
