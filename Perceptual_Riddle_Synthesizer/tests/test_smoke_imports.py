def test_spatial_package_importable():
    import spatial

    assert hasattr(spatial, 'generate_spatial_questions')
