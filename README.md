# REM

This is a ground-up rewrite of the bytecode interpreter underpinning the Souvenir programming language. Its virtual machine runs processes concurrently with message passing as the primary means of communication. Unlike Erlang, where messages must be explicitly retrieved from an inbox, REM implements an interrupt model where an incoming message temporarily suspends execution of the process's main program if a suitable message handler has been installed at the time the message arrives.

This implementation makes several important changes from the original design:

1. Execution occurs in units of basic blocks. Any time a process performs control flow (such as returning to the start of a loop) or an IO operation (such as sending a message) it surrenders control to the supervisor, which then has an opportunity to deliver a message. This allows for more useful guarantees than the original model in which messages could arrive at any time.

2. Rather than a copying garbage collector, this implementation uses atomic reference counting and allows data structures to be shared between processes. Note that data structures are immutable once created, and the memory layout is always a DAG.

3. Rather than pattern matching in the body of a message handler, a process builds a reified pattern match structure before arming the handler. This allows the supervisor to immediately filter out incompatible handlers when a message is received. Note that a handler still has an opportunity to reject a message, passing it to the next handler in sequence.

Certain functionality important for the Souvenir use case is currently unimplemented, but the fundamental concurrency model is in place.
