import typer
from rich.console import Console
import pkg_resources

__version__ = pkg_resources.get_distribution("myoquant").version
__version_cellpose__ = pkg_resources.get_distribution("cellpose").version
__version_stardist__ = pkg_resources.get_distribution("stardist").version
__version_torch__ = pkg_resources.get_distribution("torch").version
__version_tensorflow__ = pkg_resources.get_distribution("tensorflow").version

from myoquant.commands.docs import app as docs_app
from myoquant.commands import run_sdh, run_he, run_atp

console = Console()


def version_callback(value: bool):
    if value:
        print(
            f"MyoQuant Version: {__version__} \nCellpose Version: {__version_cellpose__} \nStardist Version: {__version_stardist__} \nTorch Version: {__version_torch__} \nTensorflow Version: {__version_tensorflow__}"
        )
        raise typer.Exit()


app = typer.Typer(
    name="MyoQuant",
    add_completion=False,
    help="MyoQuant Analysis Command Line Interface",
    pretty_exceptions_show_locals=False,
)
app.add_typer(docs_app, name="docs", help="Generate documentation")

app.registered_commands += (
        run_sdh.app.registered_commands
        + run_he.app.registered_commands
        + run_atp.app.registered_commands
)

@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    )
):
    """
    MyoQuant Analysis Command Line Interface
    """

if __name__ == "__main__":
    app()
