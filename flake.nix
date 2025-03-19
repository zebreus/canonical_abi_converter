{
  description = "Canonical abi converter";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs";
    nixpkgs-wit-bindgen.url = "github:nixos/nixpkgs/563c21191ff0600457bd85dc531462c073a1574b";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-wit-bindgen,
      fenix,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs-wit-bindgen = import nixpkgs-wit-bindgen {
          inherit system;
        };
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            (final: prev: {
              wit-bindgen = pkgs-wit-bindgen.wit-bindgen;
            })
            fenix.overlays.default
          ];
        };

        fenixPkgs = fenix.packages.${system};
        rustToolchain = fenixPkgs.combine [
          fenixPkgs.complete.toolchain

          (fenixPkgs.complete.withComponents [
            "cargo"
            "clippy"
            "rust-src"
            "rustc"
            "rustfmt"
          ])
        ];

      in
      {
        name = "canonical-abi-converter";

        devShell = pkgs.mkShell {
          buildInputs = [
            rustToolchain
            pkgs.rust-analyzer-nightly
            pkgs.wasm-tools
            pkgs.wit-bindgen
            pkgs.cargo-rdme
            pkgs.jq
          ];
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
