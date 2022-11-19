from typer.testing import CliRunner

from myoquant.__main__ import app

runner = CliRunner()


def test_sdh_analysis():
    result = runner.invoke(
        app, ["sdh-analysis", "sample_img/sample_sdh.jpg", "--cellpose-diameter", 80]
    )
    assert result.exit_code == 0
    assert "Analysis completed !" in result.stdout


def test_he_analysis():
    # Note that we need to confirm here, hence the extra input!
    result = runner.invoke(
        app, ["he-analysis", "sample_img/sample_he.jpg", "--cellpose-diameter", 80]
    )
    assert result.exit_code == 0
    assert "Analysis completed !" in result.stdout
