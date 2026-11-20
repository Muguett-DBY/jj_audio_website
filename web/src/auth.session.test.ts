import { describe, expect, it } from "vitest";
import { ApiError } from "./api";
import {
  ANON_SESSION_UNAVAILABLE,
  API_DOWN_HINT,
  sessionFailureHint,
  sessionFailureMessage,
} from "./auth";

describe("anonymous session failure", () => {
  it("asks the user to sign in when the server returns 404", () => {
    const message = sessionFailureMessage(new ApiError(404, "Not Found"));
    expect(message).toBe(ANON_SESSION_UNAVAILABLE);
    expect(sessionFailureHint(message)).toBe(ANON_SESSION_UNAVAILABLE);
    expect(sessionFailureHint(message)).not.toMatch(/API is running/);
  });

  it("keeps the API hint for network and server failures", () => {
    const network = sessionFailureMessage(new TypeError("Failed to fetch"));
    expect(network).toBe("Failed to fetch");
    expect(sessionFailureHint(network)).toBe(API_DOWN_HINT);

    const server = sessionFailureMessage(new ApiError(503, "Service Unavailable"));
    expect(sessionFailureHint(server)).toBe(API_DOWN_HINT);
    expect(sessionFailureHint(server)).toMatch(/API is running/);
  });
});
