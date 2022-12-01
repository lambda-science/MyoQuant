import typer
from rich.console import Console

from .commands.docs import app as docs_app
from .commands import run_sdh, run_he

console = Console()

app = typer.Typer(
    name="MyoQuant",
    add_completion=False,
    help="MyoQuant Analysis Command Line Interface",
    pretty_exceptions_show_locals=False,
)
app.add_typer(docs_app, name="docs", help="Generate documentation")


app.registered_commands += (
    run_sdh.app.registered_commands + run_he.app.registered_commands
)

if __name__ == "__main__":
    app()
