import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter } from "react-router-dom";

import App from "./App";
import "./styles/tokens.css";

// HashRouter rather than BrowserRouter: the build is deployed as static files
// (GitHub Pages, or any object store) where deep links have no server rewrite.
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>,
);
