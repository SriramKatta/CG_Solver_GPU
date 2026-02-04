# Build Script README

This project provides a simple helper script (`compile.sh`) to build and manage the project using CMake on the cluster.

---

## Usage

```
./compile.sh {compile|clean|format}
```

---

## Commands

### 1. compile

⚠️ **First run note:**
The first compilation may take longer because required libraries and dependencies are downloaded from Git and cached in "~/.cache/CPM".

```
./compile.sh compile
```

* Configures the project (if needed)
* Builds the code using parallel compilation


---

### 2. clean

```
./compile.sh clean
```

* Removes the `build/` directory
* Reconfigures and rebuilds the project from scratch

Use this if you want a fresh build.

---

### 3. format

```
./compile.sh format
```

* Runs the `fix-clang-format` target
* Automatically formats the source code

---

## Build Directory

All generated files are stored in:

```
build/
```

This keeps build files separate from source code.

---

## Notes

* Required modules and environment variables are set automatically in the script
* Parallel build is enabled by default
* Avoid frequent `clean` builds to save time

---

