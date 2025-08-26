from src.data.dataset import load_data

def test_load_data(tmp_path):
    # Create dummy CSV
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("f1,f2,label\n1,2,0\n3,4,1")
    df = load_data(csv_file)
    assert not df.empty
    assert "label" in df.columns
