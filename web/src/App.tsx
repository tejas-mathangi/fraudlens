import { Navigate, Route, Routes } from "react-router-dom";

import { AppShell } from "./components/shell/AppShell";
import { EmptyState } from "./components/ui/States";
import { ExplainPage } from "./pages/Explain";
import { ExplorerPage } from "./pages/Explorer";
import { ModelPage } from "./pages/Model";
import { NodeInspectorPage } from "./pages/NodeInspector";
import { OverviewPage } from "./pages/Overview";
import { RingsPage } from "./pages/Rings";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<OverviewPage />} />
        <Route path="explorer" element={<ExplorerPage />} />
        <Route path="node/:idx" element={<NodeInspectorPage />} />
        <Route path="rings" element={<RingsPage />} />
        <Route path="explain" element={<ExplainPage />} />
        <Route path="model" element={<ModelPage />} />
        <Route
          path="404"
          element={<EmptyState title="Page not found" description="That route does not exist." />}
        />
        <Route path="*" element={<Navigate to="/404" replace />} />
      </Route>
    </Routes>
  );
}
