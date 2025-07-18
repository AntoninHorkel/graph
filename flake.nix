{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils = {
      url = "github:numtide/flake-utils";
      # inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      # inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };
  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      crane,
      rust-overlay,
      advisory-db,
    }:
    # TODO: Specify systems.
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
        };
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        # https://github.com/ipetkov/crane/blob/master/lib/cleanCargoSource.nix
        src = pkgs.lib.cleanSourceWith {
          src = pkgs.lib.cleanSource ./.;
          # TODO: Fix regex!
          filter =
            path: type: (builtins.match ".*src\/.*" path != null) || (craneLib.filterCargoSources path type);
          name = "source";
        };
        buildInputs = with pkgs; [
          libGL
          libxkbcommon
          vulkan-loader
          # vulkan-validation-layers
          wayland
        ];
        LD_LIBRARY_PATH =
          with pkgs;
          lib.makeLibraryPath [
            libGL
            libxkbcommon
            vulkan-loader
            # vulkan-validation-layers
            wayland
          ];
        commonArgs = {
          inherit src buildInputs;
          strictDeps = true;
          RUSTFLAGS =
            with pkgs;
            lib.optionalString stdenv.hostPlatform.isLinux ("-C link-args=-Wl,-rpath=" + LD_LIBRARY_PATH);
          # TODO: Set Cargo profile the default way.
          cargoBuildCommand = "cargo build --profile $profile";
          cargoCheckCommand = "cargo check --profile $profile";
          cargoTestCommand = "cargo test --profile $profile";
          # TODO: Remove this when RUSTFLAGS starts working as expected.
          addRunpathsPhase = ''
            while IFS= read -r -d $'\0' path ; do
              patchelf "$path" --print-rpath >& /dev/null || continue
              patchelf "$path" --add-rpath ${LD_LIBRARY_PATH}
            done < <(
              for o in $(getAllOutputNames) ; do
                find "''${!o}" -type f -and "(" -executable -or -iname '*.so' ")" -print0
              done
            )
          '';
          postPhases = with pkgs; lib.optionals stdenv.hostPlatform.isLinux [ "addRunpathsPhase" ];
        };
        commonArgsDebug = commonArgs // {
          profile = "dev";
        };
        cargoArtifactsDebug = craneLib.buildDepsOnly commonArgsDebug;
        debug = craneLib.buildPackage (commonArgsDebug // { inherit cargoArtifactsDebug; });
        release = craneLib.buildPackage (commonArgs // { profile = "release"; });
      in
      {
        checks = {
          inherit debug;
          fmt = craneLib.cargoFmt {
            inherit src;
          };
          toml-fmt = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [ ".toml" ];
          };
          clippy = craneLib.cargoClippy {
            inherit src;
            cargoArtifacts = cargoArtifactsDebug;
          };
          audit = craneLib.cargoAudit {
            inherit src advisory-db;
          };
        };
        packages = {
          inherit debug release;
          default = release;
        };
        devShells.default = craneLib.devShell {
          inherit LD_LIBRARY_PATH;
          checks = self.checks.${system};
          packages = with pkgs; [
            bacon
            git
            lldb
            nixfmt-tree
            pre-commit
            tracy
            wgsl-analyzer
          ];
          shellHook = "pre-commit install";
          RUST_BACKTRACE = 1;
        };
        formatter = pkgs.nixfmt-tree;
      }
    );
}
