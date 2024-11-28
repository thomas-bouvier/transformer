{ pkgs, lib, config, inputs, ... }:

{
  # https://github.com/cachix/devenv/issues/1264#issuecomment-2368362686
  packages = with pkgs; [
    graphviz
    gcc-unwrapped # fix: libstdc++.so.6: cannot open shared object file
    libz # fix: for numpy/pandas import
  ];

  env.LD_LIBRARY_PATH = lib.mkIf pkgs.stdenv.isLinux (
    lib.makeLibraryPath (with pkgs; [
      zlib
      gcc-unwrapped.lib
      stdenv.cc.cc.lib # required by jupyter
    ])
  );

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.12";

    uv = {
      enable = true;
      sync = {
        enable = true;
        arguments = ["--frozen"];
      };
    };
  };

  env.UV_PYTHON = config.languages.python.package;
  env.VENV_PATH = "${config.env.DEVENV_STATE}/venv";

  enterShell = ''
    python_version="${config.languages.python.package.version}"
    echo "Python version: $python_version"
    echo "UV version: $(uv version)"
    echo "Virtual environment: $VENV_PATH"

    echo
    echo "Source $VENV_PATH/bin/activate and you are good :)"
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
