import { Sidebar } from './components/Sidebar';
import { ChatView } from './components/ChatView';

export default function App() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <ChatView />
      </main>
    </div>
  );
}
