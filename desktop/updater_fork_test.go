package main

import "testing"

func TestForkBuildDoesNotOfferOfficialReasonixUpdates(t *testing.T) {
	oldRepository, oldVersion := releaseRepository, version
	releaseRepository = "Junjie88/Reasionix-SupportVisionModel"
	version = "v1.19.6-vision.1"
	t.Cleanup(func() {
		releaseRepository = oldRepository
		version = oldVersion
	})

	info, err := (&App{}).CheckUpdate("stable")
	if err != nil {
		t.Fatalf("CheckUpdate: %v", err)
	}
	if info.Available {
		t.Fatal("fork build must not offer an official Reasonix update")
	}
	if !info.ManualOnly {
		t.Fatal("fork build must use its repository release page")
	}
	if got, want := info.DownloadURL, "https://github.com/Junjie88/Reasionix-SupportVisionModel/releases"; got != want {
		t.Fatalf("DownloadURL = %q, want %q", got, want)
	}
}

func TestOfficialBuildKeepsOfficialUpdateSource(t *testing.T) {
	oldRepository := releaseRepository
	releaseRepository = "esengine/DeepSeek-Reasonix"
	t.Cleanup(func() { releaseRepository = oldRepository })

	if !usesOfficialReleaseRepository() {
		t.Fatal("official repository must keep the normal updater")
	}
	if got, want := releasePageURL(), "https://github.com/esengine/DeepSeek-Reasonix/releases"; got != want {
		t.Fatalf("releasePageURL = %q, want %q", got, want)
	}
}
