from typer.testing import CliRunner

from myoquant.__main__ import app

runner = CliRunner()


def test_sdh_analysis():
    result = runner.invoke(
        app,
        [
            "sdh-analysis",
            "sample_img/sample_sdh.jpg",
            "--cellpose-diameter",
            80,
            "--mask-path",
            "sample_img/binary_mask_sdh.tif",
        ],
    )
    assert result.exit_code == 0
    assert "Analysis Results" in result.stdout


def test_he_analysis():
    result = runner.invoke(
        app,
        [
            "he-analysis",
            "sample_img/sample_he.jpg",
            "--cellpose-diameter",
            80,
            "--mask-path",
            "sample_img/binary_mask.tif",
        ],
    )
    assert result.exit_code == 0
    assert "Analysis Results" in result.stdout


def test_he_analysis_fluo():
    result = runner.invoke(
        app,
        [
            "he-analysis",
            "sample_img/cytoplasm.tif",
            "--cellpose-diameter",
            80,
            "--mask-path",
            "sample_img/binary_mask_fluo.tif",
            "--fluo-nuc",
            "sample_img/nuclei.tif",
        ],
    )
    assert result.exit_code == 0
    assert "Analysis Results" in result.stdout


def test_atp_analysis():
    result = runner.invoke(
        app,
        [
            "atp-analysis",
            "sample_img/sample_atp.jpg",
            "--cellpose-diameter",
            80,
        ],
    )
    assert result.exit_code == 0
    assert "Analysis Results" in result.stdout
