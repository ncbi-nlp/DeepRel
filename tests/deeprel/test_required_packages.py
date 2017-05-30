import pkg_resources


def test_required_packages():
    dependencies = []
    with open('requirements.txt') as fp:
        for line in fp:
            dependencies.append(line.strip())

    pkg_resources.require(dependencies)


if __name__ == '__main__':
    test_required_packages()