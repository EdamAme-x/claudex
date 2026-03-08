import esbuild from "esbuild";
import { writeFileSync } from "fs";

// react-devtools-core を空モジュールに置き換えるplugin
const stubPlugin = {
  name: "stub-optional",
  setup(build) {
    build.onResolve({ filter: /^react-devtools-core$/ }, () => ({
      path: "react-devtools-core",
      namespace: "stub",
    }));
    build.onLoad({ filter: /.*/, namespace: "stub" }, () => ({
      contents: "export default {};",
      loader: "js",
    }));
  },
};

// CJS互換のrequireをバンドル先頭で定義
// ink等のCJS依存がrequire("assert")等を使うため
const banner = `
import { createRequire as _cr } from "module";
var require = _cr(import.meta.url);
`.trim();

await esbuild.build({
  entryPoints: ["src/index.tsx"],
  bundle: true,
  outfile: "dist/claudex.mjs",
  platform: "node",
  target: "node18",
  format: "esm",
  jsx: "automatic",
  plugins: [stubPlugin],
  banner: { js: banner },
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});

// シェバン付きのentrypoint
writeFileSync(
  "dist/claudex.js",
  `#!/usr/bin/env node
import("./claudex.mjs").catch(e => { console.error(e.message); process.exit(1); });
`
);

console.log("Built dist/claudex.mjs + dist/claudex.js");
